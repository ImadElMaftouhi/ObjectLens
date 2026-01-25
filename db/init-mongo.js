// MongoDB initialization script for ObjectLens
// Creates database, collections, and indexes for image metadata

const dbName = process.env.MONGO_INITDB_DATABASE || "objectlens";
db = db.getSiblingDB(dbName);

print(`Initializing database: ${dbName}`);

// Collection: images
// Stores metadata about images and their detected objects
// Images themselves are stored on disk/mounted volumes, not in MongoDB
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
                            required: ["object_idx", "faiss_id", "bbox", "class_id", "class_name"],
                            properties: {
                                object_idx: { bsonType: "int" },
                                faiss_id: { bsonType: "int" },  // Reference to FAISS index
                                bbox: {
                                    bsonType: "array",
                                    items: { bsonType: "int" },
                                    minItems: 4,
                                    maxItems: 4
                                },
                                class_id: { bsonType: "int" },
                                class_name: { bsonType: "string" },
                                confidence: { bsonType: "double" }
                            }
                        }
                    },
                    num_objects: { bsonType: "int" },
                    indexed_at: { bsonType: "date" }
                }
            }
        }
    });
    
    print(" Created 'images' collection");
}

// Collection: objects (optional - for direct object queries)
// Stores individual objects with FAISS ID references
if (!db.getCollectionNames().includes("objects")) {
    db.createCollection("objects", {
        validator: {
            $jsonSchema: {
                bsonType: "object",
                required: ["_id", "faiss_id", "image_path", "object_idx"],
                properties: {
                    _id: { bsonType: "string" },  // Composite: image_path__object_idx
                    faiss_id: { bsonType: "int" },
                    image_path: { bsonType: "string" },
                    object_idx: { bsonType: "int" },
                    bbox: {
                        bsonType: "array",
                        items: { bsonType: "int" }
                    },
                    class_id: { bsonType: "int" },
                    class_name: { bsonType: "string" },
                    confidence: { bsonType: "double" }
                }
            }
        }
    });
    
    print(" Created 'objects' collection");
}

// Create indexes for fast queries
print(" Creating indexes...");

// Images collection indexes
db.images.createIndex({ "image_path": 1 }, { unique: true });
db.images.createIndex({ "split": 1 });
db.images.createIndex({ "objects.class_id": 1 });
db.images.createIndex({ "objects.class_name": 1 });
db.images.createIndex({ "objects.faiss_id": 1 }, { sparse: true });

// Objects collection indexes
db.objects.createIndex({ "faiss_id": 1 }, { unique: true });
db.objects.createIndex({ "image_path": 1 });
db.objects.createIndex({ "object_idx": 1 });
db.objects.createIndex({ "class_id": 1 });
db.objects.createIndex({ "class_name": 1 });

print(" Indexes created");

// Collection: index_metadata
// Stores metadata about the FAISS index itself
if (!db.getCollectionNames().includes("index_metadata")) {
    db.createCollection("index_metadata");
    
    // Store FAISS index metadata
    db.index_metadata.insertOne({
        _id: "faiss_index",
        num_vectors: 0,
        dimension: 0,
        metric: "cosine",
        index_type: "IndexFlatIP",
        created_at: new Date(),
        vectors_file: "vectors.npy",
        ids_file: "ids.npy",
        index_file: "index.faiss"
    });
    
    print(" Created 'index_metadata' collection");
}

print(`\n Database '${dbName}' initialized successfully!`);
print("   Collections: images, objects, index_metadata");
print("   Ready to load feature vectors and metadata\n");
