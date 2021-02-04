// Ignacio Vellido Expósito

// - Ejercicio 1: Crear en vuestra base de datos MongoDB la colección "restaurants" desde el archivo /var/tmp/restaurantes1.json conforme se indica en la transparencia 44 de la presentación sobre NoSQL. Elaborar el código MapReduce que resuelva la consulta:
// "Obtener, para el barrio "Bronx", el par de restaurantes más próximos para cada "zipcode", mostrando el barrio, el nombre, la dirección, la distancia entre ellos y la cantidad de restaurantes evaluados para cada "zipcode", para aquellos restaurantes que hayan tenido un "score" mayor o igual que 11 en alguna ocasión".

db.runCommand({
    mapReduce: "restaurants",
    // key = zipcode
    // value = {contadorRestaurantes, infoRestaurante}
    map: function map() {
        emit(
            this.address.zipcode, 
            {
                "Contador": 1,
                "Info" : {
                    "Nombre": this.name, 
                    "Direccion": this.address
                }
            }
        );
    },
    reduce: function reduce(key, values) {
        reduced = {"Contador": 0, "Info": []};
        
        for (c in values) {
            reduced.Contador += values[c].Contador;
            reduced.Info.push(values[c].Info);
        }

        return reduced;
    },
    // Coger restaurantes a distancia mínima
    finalize: function finalize(key, reduced) {
        if (reduced.Contador == 1) {
            return { "message" : "Zipcode con un solo restaurante" };
        }
        
        var min_dist = 999999999999;
        var rest1 = { "Nombre": "", "Direccion": {} };
        var rest2 = { "Nombre": "", "Direccion": {} };
        var c1, c2, d2;
        
        for (var i in reduced.Info) {
            for (var j in reduced.Info) {
                if (i>=j) continue; //termina la iteración actual y continua con la siguiente j

                c1 = reduced.Info[i].Direccion.coord;
                c2 = reduced.Info[j].Direccion.coord;
                d2 = (c1[0]-c2[0])*(c1[0]-c2[0])+(c1[1]-c2[1])*(c1[1]-c2[1]);

                if (d2 < min_dist && d2 > 0) {
                    min_dist = d2;
                    rest1 = reduced.Info[i];
                    rest2 = reduced.Info[j];
                }
            }
        }
    
        return{
            "Barrio": "Bronx",
            "Restaurantes Evaluados": reduced.Contador,
            "Restaurante 1": rest1,
            "Restaurante 2": rest2,
            "Distancia entre ellos": min_dist
        };
    },
    // Nos quedamos solo con los del barrio Bronx de score >= 11
    query: { "borough": "Bronx", "grades.score": { $gt: 11 } },
    
    out: {inline: 1}
})

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// - Ejercicio 2: Resolver la pregunta anterior usando un enfoque basado en el uso del operador aggregate.

db.runCommand({
    aggregate: "restaurants",
    pipeline:[        
        {$match: {  // Nos quedamos con los del barrio Bronx de score >= 11
            "borough": "Bronx",
            "grades.score": { $gt: 11 }
        }},

        // Agrupamos por zipcode y nos quedamos con los campos relevantes
        {$group: {
            _id: "$address.zipcode",
            CantidadRestaurantes: { $sum: 1 },
            restaurante1: {$push: {
                info: {
                    Nombre: "$name",
                    Direccion: "$address"
                },
                lat: {$arrayElemAt: ["$address.coord", 0]},
                long: {$arrayElemAt: ["$address.coord", 1]}
            }},
            restaurante2: {$push: {
                info: {
                    Nombre: "$name",
                    Direccion: "$address"
                },
                lat: {$arrayElemAt: ["$address.coord", 0]},
                long: {$arrayElemAt: ["$address.coord", 1]}
            }},
        }},

        // Desanidamos para tener parejas de todos contra todos
        {$unwind: "$restaurante1"},
        {$unwind: "$restaurante2"},

         //Calcula la distancia entre cada par
        {$project: {
            _id: 0,
            zipcode: "$_id",
            CantidadRestaurantes: "$CantidadRestaurantes",
            restaurante1: "$restaurante1.info",
            restaurante2: "$restaurante2.info",
            distancia:{ $sqrt: {$sum: [ {$pow: [{$subtract: ["$restaurante1.lat","$restaurante2.lat"]},   2]},
                                        {$pow: [{$subtract: ["$restaurante1.long","$restaurante2.long"]}, 2]}
                                    ]
                        }}
        }},

        // Eliminamos parejas redundantes y aquellas a distancia 0 (sobre sí misma).
        {$redact: {"$cond": [{$and:[{"$lt": ["$restaurante1.Nombre", "$restaurante2.Nombre"]},{"$ne":["$distancia",0.0]}]},"$$KEEP","$$PRUNE"]}},

        // Volvemos a agrupar por zipcode
        {$group: {
            _id: "$zipcode",
            CantidadRestaurantes: { $min: "$CantidadRestaurantes" },    // El array contiene el mismo valor
            "dist_min": {$min: "$distancia"}, // Obtenemos las distancia mínima para cada país
            
            "parejas": {$push: {
                restaurante1: "$restaurante1", 
                restaurante2: "$restaurante2", 
                distancia: "$distancia"
            }}
        }},

        {$unwind: "$parejas"}, // Desanidamos las parejas

        // Cogemos pareja a distancia mínima
        {$redact: {"$cond": [{"$eq": ["$dist_min", "$parejas.distancia"]}, "$$KEEP", "$$PRUNE"]}},

        // Proyectamos los datos pedidos
        {$project: {
            _id: 0,
            Barrio: "Bronx",
            CantidadRestaurantes: { $toInt : "$CantidadRestaurantes" },
            Restaurantes: "$parejas",
        }}
    ],
    allowDiskUse: true,
    cursor: { batchSize: 10 }
})