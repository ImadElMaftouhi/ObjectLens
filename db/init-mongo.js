// db/init-mongo.js
const dbName = process.env.MONGO_INITDB_DATABASE || "objectlens"
db = db.getSiblingDB(dbName)

// Single collection: images (each document = one image with objects[])
if (!db.getCollectionNames().includes("images")) {
  db.createCollection("images", {
    validator: {
      $jsonSchema: {
        bsonType: "object",
        required: ["_id", "image_path", "objects"],
        properties: {
          _id: { bsonType: "string" },
          image_path: { bsonType: "string" },
          split: { bsonType: "string" },
          width: { bsonType: "int" },
          height: { bsonType: "int" },
          objects: {
            bsonType: "array",
            items: {
              bsonType: "object",
              required: ["bbox", "class_id", "final_vector"],
              properties: {
                bbox: {
                  bsonType: "array",
                  items: { bsonType: "int" },
                  minItems: 4,
                  maxItems: 4
                },
                class_id: { bsonType: "int" },
                class_name: { bsonType: "string" },
                confidence: { bsonType: "double" },
                features: { bsonType: "object" }, // keep flexible
                final_vector: {
                  bsonType: "array",
                  items: { bsonType: "double" }
                },
                vector_dim: { bsonType: "int" }
              }
            }
          }
        },
        additionalProperties: true
      }
    }
  })
}

// Helpful indexes
db.images.createIndex({ image_path: 1 })
db.images.createIndex({ "objects.class_id": 1 })
