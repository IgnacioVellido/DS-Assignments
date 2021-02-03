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

// - Ejercicio 2: Resolver la pregunta anterior usando un enfoque basado en el uso del operador aggregate.

db.runCommand({
    aggregate: "restaurants",
    pipeline:[        
        {$match: {  // Nos quedamos con los del barrio Bronx
            "borough": "Bronx",
            // "grades"
        }},
        {$group: {
            _id: "$address.street",
            CantidadRestaurantes: { $sum: 1 },
            restaurantes: {$push: "$name"}
        }},        
        {$sort: {   // Ordenar de forma decreciente
            CantidadRestaurantes: -1
        }},
        {$project: {
            _id: 0,
            Calle: "$_id",
            CantidadRestaurantes: { $toInt : "$CantidadRestaurantes" },
            Restaurantes: "$restaurantes"
        }}
    ],
    cursor: { batchSize: 1 }    // Quedarnos con el primero
})