// 1. Resolver mediante MapReduce y agregación la siguiente consulta: “Mostrar, para cada calle, cuantos restaurantes hay”.

// MapReduce -  0.3s
db.runCommand({
    mapReduce: "restaurants",
    map: function map() {
        // key = Nombre de la calle
        // value = 1
        emit(
            this.address.street, 1
        );
    },
    reduce: function reduce(calle, contadores) {
        reduced = 0;
        
        for (c in contadores) {
            reduced += contadores[c];
        }

        return reduced;
    },
    out: {inline: 1}
})

// Agregación - 0.066s
db.runCommand({
    aggregate: "restaurants",
    pipeline:[
        {$group: {      // Agrupar por calle y sumar por cada instancia
            _id: "$address.street",
            CantidadRestaurantes: { $sum: 1 }
        }},
        {$project: {    // Se redondea el número de restaurantes
            _id: 0,
            Calle: "$_id",
            CantidadRestaurantes: { $toInt : "$CantidadRestaurantes" }
        }}
    ],
    cursor: { batchSize: 100 }
})

// -----------------------------------------------------------------------------
// Para la consulta anterior, considerar sólo los restaurantes que hayan obtenido un grado “A” en cualquier momento.

// MapReduce -  0.292s
db.runCommand({
    mapReduce: "restaurants",
    map: function map() {
        // key = Nombre de la calle
        // value = 1
        emit(
            this.address.street, 1
        );
    },
    reduce: function reduce(calle, contadores) {
        reduced = 0;
        
        for (c in contadores) {
            reduced += contadores[c];
        }

        return reduced;
    },
    // Nos quedamos solo con los de nota A
    query: { "grades.grade": "A" },
    out: {inline: 1}
})

// Aggregate - 0.067s
db.runCommand({
    aggregate: "restaurants",
    pipeline:[        
        {$match: {  // Nos quedamos solo con los de nota A
            "grades.grade": "A"
        }},
        {$group: {
            _id: "$address.street",
            CantidadRestaurantes: { $sum: 1 },
        }},
        {$project: {
            _id: 0,
            Calle: "$_id",
            CantidadRestaurantes: { $toInt : "$CantidadRestaurantes" },
        }}
    ],
    cursor: { batchSize: 100 }
})

// -----------------------------------------------------------------------------
// Para la consulta anterior, muestra el resultado ordenado, de forma decreciente, por cantidad de restaurantes.

// MapReduce - 0.368s
db.runCommand({
    mapReduce: "restaurants",
    map: function map() {
        // key = Nombre de la calle
        // value = 1
        emit(
            this.address.street, 1
        );
    },
    reduce: function reduce(calle, contadores) {
        reduced = 0;
        
        for (c in contadores) {
            reduced += contadores[c];
        }

        return reduced;
    },
    // Nos quedamos solo con los de nota A
    query: { "grades.grade": "A" },
    // Guardamos en una colección
    out: {replace: "resultados"}
})

// Ordenamos el documento de salida
db.resultados.find().sort({"value": -1})

// Aggregate - 0.08s
db.runCommand({
    aggregate: "restaurants",
    pipeline:[        
        {$match: {  // Nos quedamos solo con los de nota A
            "grades.grade": "A"
        }},
        {$group: {
            _id: "$address.street",
            CantidadRestaurantes: { $sum: 1 },
        }},
        {$sort: {   // Ordenar de forma decreciente
            CantidadRestaurantes: -1
        }},
        {$project: {
            _id: 0,
            Calle: "$_id",
            CantidadRestaurantes: { $toInt : "$CantidadRestaurantes" },
        }}
    ],
    cursor: { batchSize: 1000 }
})

// -----------------------------------------------------------------------------
// Para la consulta del ejercicio 2, muestra la calle y los nombres de los restaurantes de la calle que más restaurantes tenga.

// MapReduce - 0.564s
db.runCommand({
    mapReduce: "restaurants",
    map: function map() {
        // key = Nombre de la calle
        // value = 1
        emit(
            this.address.street, {"Contador": 1, "Restaurantes": this.name}
        );
    },
    reduce: function reduce(calle, contadores) {
        reduced = {"Contador": 0, "Restaurantes":[]};
        
        for (c in contadores) {
            reduced.Contador += contadores[c].Contador;
            reduced.Restaurantes.push(contadores[c].Restaurantes);
        }

        return reduced;
    },
    // Nos quedamos solo con los de nota A
    query: { "grades.grade": "A" },
    // Guardamos en una colección
    out: {replace: "resultados"}
})

// Ordenamos y nos quedamos con el primero
db.resultados.find().sort({"value": -1}).limit(1)

// Aggregate - 0.079s
db.runCommand({
    aggregate: "restaurants",
    pipeline:[        
        {$match: {  // Nos quedamos solo con los de nota A
            "grades.grade": "A"
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

// -----------------------------------------------------------------------------
// Podéis comprobar que el tiempo de ejecución mediante el enfoque de agregación 
// es unas 10 veces menor que mediante el enfoque de MapReduce, además de que, en 
// algunos ejercicios, el enfoque 
// MapReduce precisa de la creación de un relación intermedia.
// ¿Qué concluís acerca de que enfoque es más adecuado para cada tipo de   
// problema?

// Depende en gran medida del número de instancias sobre el que se trabaja.
// Aunque la ejecución del modelo MapReduce puede ser más rápida, la carga que
// genera la separación en clave-valor y control de las operaciones map-reduce
// puede generar más carga que el pipeline de agregación para un conjunto de
// datos pequeños.