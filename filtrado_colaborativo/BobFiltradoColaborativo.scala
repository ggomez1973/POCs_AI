
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{array, lit, map, struct}
import org.apache.log4j._
import scala.math.sqrt

// Bought also Bought (BoB) con filtrado colaborativo
object Bob {
  
  type ProductRating = (Int, Double)
  type UserRatingPair = (Int, (ProductRating, ProductRating))
  
  def armarTuplas(userRatings:UserRatingPair) = {
    val productRating1 = userRatings._2._1
    val productRating2 = userRatings._2._2
    
    val product1 = productRating1._1
    val rating1 = productRating1._2
    val product2 = productRating2._1
    val rating2 = productRating2._2
    
    ((product1, product2), (rating1, rating2))
  }
  
  def filtrarDuplicados(userRatings:UserRatingPair):Boolean = {
    val productRating1 = userRatings._2._1
    val productRating2 = userRatings._2._2
    
    val product1 = productRating1._1
    val product2 = productRating2._1
    
    return product1 < product2
  }
  
  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]
  
  // Similitud existente entre dos vectores en un espacio
  def calcularSimilitudCoseno(ratingPairs:RatingPairs): (Double, Int) = {
    var numPairs:Int = 0
    var sum_xx:Double = 0.0
    var sum_yy:Double = 0.0
    var sum_xy:Double = 0.0
    
    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2
      
      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingY
      numPairs += 1
    }
    
    val numerador:Double = sum_xy
    val denominador = sqrt(sum_xx) * sqrt(sum_yy)
    
    var score:Double = 0.0
    if (denominador != 0) {
      score = numerador / denominador
    }
    
    return (score, numPairs)
  }
  
  def main(args: Array[String]) {
    // Seteo el logger para ver solo errores.
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Parsing y formateo para llegar al formato de tuplas necesario para aplicar Collaborative filtering
    // Creo una sesion con un solo servidor de Spark local
    val spark = SparkSession.builder().appName("Demo BoB").config("spark.master", "local").getOrCreate()
    
    // Creo un dataframe para los productos
    val df_products = spark.read.option("header", true).option("inferSchema", true).csv("/home/german/codigo/demobobsets/products.csv")
    df_products.show()
    
    // Creo un dataframe para las ordenes (carrito)
    val df_orders = spark.read.option("header", true).option("inferSchema", true).csv("/home/german/codigo/demobobsets/orders.csv")
    
    // Creo un dataframe para los productos por orden
    val df_order_products = spark.read.option("header", true).option("inferSchema", true).csv("/home/german/codigo/demobobsets/order_products.csv")
    
    // Armo mi dataframe para trabajar
    val df_join = df_orders.join(df_order_products, df_orders.col("order_id") === df_order_products.col("order_id"))
    val df = df_join.groupBy("user_id", "product_id").count()
    val df_quantity = df.withColumn("cantidad", df.col("count"))
   
    // Feature scaling para poner los valores entre 0 y 1 (score)
    // Como no tengo score, lo calculo basado en la cantidad de veces que un producto fue comprado.
    // Para eso necesito saber la cantidad de ventas de cada uno.
    val max = df_quantity.groupBy("product_id").max("count").collect(){0}.getLong(1)
    val min = 1    
    val df_scaled = df_quantity.withColumn("score", lit(df_quantity.col("cantidad").-(min)./(max-min)))

    // Formato para Collaborative Filtering
    val rdd_mapped = df_scaled.select("user_id","product_id", "score").rdd
    val ratings = rdd_mapped.map(l => (l(0).asInstanceOf[Int], (l(1).asInstanceOf[Int], l(2).asInstanceOf[Double])))
    println("Ratings")
    ratings.foreach(println)
    
    // Armar cada par de productos ranqueados por el mismo usuario.
    // Self-join para encontrar todas las posibles combinaciones.
    val joinedRatings = ratings.join(ratings)
    println("Ratings joineados con si mismos")
    joinedRatings.foreach(println)
    // Ahora mi RDD tiene userID => ((productID, rating), (productID, rating))

    // Filtrar los pares duplicados
    val uniqueJoinedRatings = joinedRatings.filter(filtrarDuplicados)
    println("Ratings sin duplicados")
    uniqueJoinedRatings.foreach(println)
    
    // Ahora hago pares con clave (product1, product2)
    val productPairs = uniqueJoinedRatings.map(armarTuplas)
    println("Claves producto-producto")
    productPairs.foreach(println)
    
    // Ahora tenemos (product1, product2) => (rating1, rating2)
    // Ahora junto todos los ratings para cada producto y calculamos similitud
    val productPairRatings = productPairs.groupByKey()
    println("Ratings agrupadas por clave")
    productPairRatings.foreach(println)
    
    // Ahora tenemos (product1, product2) = > (rating1, rating2), (rating1, rating2) ...
    // Podemos calcular las similitudes.
    val paresProductosSimilares = productPairRatings.mapValues(calcularSimilitudCoseno).cache()
    println("Pares de similitudes")
    paresProductosSimilares.foreach(println)
    
    //Aca podria guardar los resultados si quisiera
    //val sorted = paresProductosSimilares.sortByKey()
    //sorted.saveAsTextFile("productos-similares")
    
    // De todas las recomendaciones me quedo con las mejores.
    if (args.length > 0) {
      val limiteCalidad = 0.90 // Calidad
      val limiteCoocurrencia = 1.0  // que al menos lo haya comprado 1 persona (esta muy bajo porque no tengo datos)
      
      // El producto sobre el que busco recomendaciones  
      val productId:Int = args(0).toInt
      
      // Filtro las recomendaciones segun algun criterio de calidad y co-ocurrencia.     
      val recomendaciones = paresProductosSimilares.filter( x =>
        {
          val pair = x._1
          val sim = x._2
         
          (pair._1 == productId || pair._2 == productId) && sim._1 > limiteCalidad && sim._2 > limiteCoocurrencia
        }
      )
      println("Resultados filtrados")
      recomendaciones.foreach(println)
      // Sort por calidad de score.
      val results = recomendaciones.map( x => (x._2, x._1)).sortByKey(false).take(5)
      println("Resultados")
      println(results.length)
      
      println("\nLos que compraron Mantecol(1) tambien compraron:")
      for (result <- results) {
        val sim = result._1
        val pair = result._2
        // Muestro las recomendaciones que no son del producto que estoy pidiendo.
        var productoSimilarID = pair._1
        if (productoSimilarID == productId) {
          productoSimilarID = pair._2
        }  
        import spark.implicits._
        val x = df_products.filter($"product_id" === productoSimilarID)
        x.show()  
        println("\tCalidad: " + sim._1 + "\tCantidad: " + sim._2)
      }
    }
          
    spark.stop()
  }
}