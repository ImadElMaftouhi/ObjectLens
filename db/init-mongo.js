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
              required: ["bbox", "class_id", "features"],
              properties: {
                object_id: { bsonType: "int" }, // ✅ added

                bbox: {
                  bsonType: "array",
                  items: { bsonType: "int" },
                  minItems: 4,
                  maxItems: 4
                },

                class_id: { bsonType: "int" },
                class_name: { bsonType: "string" },

                confidence: {
                  oneOf: [{ bsonType: "double" }, { bsonType: "int" }] // ✅ safer
                },

                // Per-category features (combined vectors inside)
                features: {
                  bsonType: "object",
                  required: ["form", "texture", "color"],
                  properties: {
                    form: {
                      bsonType: "object",
                      required: ["combined"],
                      properties: {
                        combined: {
                          bsonType: "array",
                          items: { bsonType: "double" }
                        }
                      },
                      additionalProperties: true
                    },
                    texture: {
                      bsonType: "object",
                      required: ["combined"],
                      properties: {
                        combined: {
                          bsonType: "array",
                          items: { bsonType: "double" }
                        }
                      },
                      additionalProperties: true
                    },
                    color: {
                      bsonType: "object",
                      required: ["combined"],
                      properties: {
                        combined: {
                          bsonType: "array",
                          items: { bsonType: "double" }
                        }
                      },
                      additionalProperties: true
                    }
                  },
                  additionalProperties: true
                }
              },
              additionalProperties: true
            }
          }
        },
        additionalProperties: true
      }
    }
  })
}

// Helpful indexes (yours are correct)
db.images.createIndex({ image_path: 1 })
db.images.createIndex({ "objects.class_id": 1 })
db.images.createIndex({ "objects.class_name": 1 })
