// db/init-mongo.js

// Use the database defined in MONGO_INITDB_DATABASE or fallback
const dbName = process.env.MONGO_INITDB_DATABASE || "objectlens"
db = db.getSiblingDB(dbName)

// =====================
// images collection
// =====================
if (!db.getCollectionNames().includes("images")) {
  db.createCollection("images", {
    validator: {
      $jsonSchema: {
        bsonType: "object",
        required: ["_id", "imagePath"],
        properties: {
          _id: {
            bsonType: "string",
            description: "imageId as string"
          },
          imagePath: {
            bsonType: "string",
            description: "Path to image as seen by backend container"
          }
        },
        additionalProperties: false
      }
    }
  })
}

// =====================
// objects collection
// =====================
if (!db.getCollectionNames().includes("objects")) {
  db.createCollection("objects", {
    validator: {
      $jsonSchema: {
        bsonType: "object",
        required: ["_id", "imageId", "features"],
        properties: {
          _id: {
            bsonType: "string",
            description: "objectId as string"
          },
          imageId: {
            bsonType: "string",
            description: "references images._id"
          },
          features: {
            bsonType: "object",
            required: ["vector", "dim"],
            properties: {
              vector: {
                bsonType: "array",
                items: { bsonType: "double" }, // numbers
                description: "L2-normalized concatenated feature vector"
              },
              dim: {
                bsonType: "int",
                description: "dimension of vector (length)"
              },
              parts: {
                bsonType: "object",
                required: [], // all optional
                properties: {
                  color: {
                    bsonType: "array",
                    items: { bsonType: "double" }
                  },
                  texture: {
                    bsonType: "array",
                    items: { bsonType: "double" }
                  },
                  shape: {
                    bsonType: "array",
                    items: { bsonType: "double" }
                  }
                },
                additionalProperties: true
              }
            },
            additionalProperties: false
          }
        },
        additionalProperties: false
      }
    }
  })

  // useful index: fast lookup by imageId
  db.objects.createIndex({ imageId: 1 })
}
