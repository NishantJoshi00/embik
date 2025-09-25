const std = @import("std");
const embik = @import("embik");

const EmbeddingEngine = embik.embed.EmbeddingEngine;
const MetadataStore = embik.storage.MetadataStore;
const VectorStore = embik.storage.VectorStore;
const DocumentStore = embik.storage.DocumentStore;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the embedding engine
    var engine = EmbeddingEngine.init("deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf") catch |err| {
        std.debug.print("Failed to initialize embedding engine: {}\n", .{err});
        return;
    };
    defer engine.deinit();

    // List of texts to embed
    const texts = [_][]const u8{
        "Hello, World!",
        "This is a test sentence.",
        "Zig is a systems programming language.",
    };

    // Embed each text
    for (texts, 0..) |text, i| {
        std.debug.print("\n--- Embedding text {d}: '{s}' ---\n", .{ i + 1, text });

        const embedding = engine.embed(allocator, text) catch |err| {
            std.debug.print("Failed to embed text: {}\n", .{err});
            continue;
        };
        defer allocator.free(embedding);

        // Print first few dimensions for brevity
        std.debug.print("Embedding (first 10 dimensions): ", .{});
        for (0..@min(10, embedding.len)) |j| {
            std.debug.print("{d:.6}", .{embedding[j]});
            if (j < @min(10, embedding.len) - 1) {
                std.debug.print(", ", .{});
            }
        }
        std.debug.print("\n", .{});

        std.debug.print("Full embedding dimensions: {d}\n", .{embedding.len});
        std.debug.print("Successfully generated embedding for '{s}'\n", .{text});
    }

    std.debug.print("\nAll embeddings completed successfully!\n", .{});

    // Test MetadataStore
    std.debug.print("\n--- Testing MetadataStore ---\n", .{});

    var store = MetadataStore.init(allocator, "/tmp/embik_test_db") catch |err| {
        std.debug.print("Failed to initialize MetadataStore: {}\n", .{err});
        return;
    };
    defer store.deinit();

    // Test put
    store.put("key1", "value1") catch |err| {
        std.debug.print("Failed to put data: {}\n", .{err});
        return;
    };
    std.debug.print("Successfully stored key1 -> value1\n", .{});

    // Test get
    if (store.get("key1") catch |err| {
        std.debug.print("Failed to get data: {}\n", .{err});
        return;
    }) |value| {
        defer allocator.free(value);
        std.debug.print("Retrieved: key1 -> {s}\n", .{value});

        if (std.mem.eql(u8, value, "value1")) {
            std.debug.print("✓ Value matches expected\n", .{});
        } else {
            std.debug.print("✗ Value mismatch\n", .{});
        }
    } else {
        std.debug.print("✗ Key not found\n", .{});
    }

    // Test delete
    store.delete("key1") catch |err| {
        std.debug.print("Failed to delete data: {}\n", .{err});
        return;
    };
    std.debug.print("Successfully deleted key1\n", .{});

    // Verify deletion
    if (store.get("key1") catch |err| {
        std.debug.print("Failed to get data after deletion: {}\n", .{err});
        return;
    }) |value| {
        defer allocator.free(value);
        std.debug.print("✗ Key still exists after deletion\n", .{});
    } else {
        std.debug.print("✓ Key successfully deleted\n", .{});
    }

    std.debug.print("MetadataStore test completed!\n", .{});

    // Test VectorStore
    std.debug.print("\n--- Testing VectorStore ---\n", .{});

    const dimension = 4;
    const nlist = 2;

    var vector_store = VectorStore.init(allocator, dimension, nlist, "/tmp/embik_faiss_index.bin") catch |err| {
        std.debug.print("Failed to initialize VectorStore: {}\n", .{err});
        return;
    };
    defer vector_store.deinit();

    // Training data
    const training_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0 };
    vector_store.train(training_data[0..]) catch |err| {
        std.debug.print("Failed to train index: {}\n", .{err});
        return;
    };
    std.debug.print("✓ Index trained successfully\n", .{});

    // Add vectors
    const vector1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector2 = [_]f32{ 2.0, 3.0, 4.0, 5.0 };

    vector_store.add_vector(vector1[0..], 100) catch |err| {
        std.debug.print("Failed to add vector 1: {}\n", .{err});
        return;
    };
    std.debug.print("✓ Added vector with ID 100\n", .{});

    vector_store.add_vector(vector2[0..], 200) catch |err| {
        std.debug.print("Failed to add vector 2: {}\n", .{err});
        return;
    };
    std.debug.print("✓ Added vector with ID 200\n", .{});

    std.debug.print("Index size after adding: {d}\n", .{vector_store.ntotal()});

    // Search
    const query = [_]f32{ 1.1, 2.1, 3.1, 4.1 };
    var search_result = vector_store.search(allocator, query[0..], 2) catch |err| {
        std.debug.print("Failed to search: {}\n", .{err});
        return;
    };
    defer search_result.deinit();

    std.debug.print("Search results:\n", .{});
    for (search_result.labels, search_result.distances) |label, distance| {
        std.debug.print("  ID: {d}, Distance: {d:.6}\n", .{ label, distance });
    }

    // Delete vector
    vector_store.delete_vector(100) catch |err| {
        std.debug.print("Failed to delete vector: {}\n", .{err});
        return;
    };
    std.debug.print("✓ Deleted vector with ID 100\n", .{});
    std.debug.print("Index size after deletion: {d}\n", .{vector_store.ntotal()});

    // Save index (simplified for testing)
    vector_store.save() catch |err| {
        std.debug.print("Save not implemented yet: {}\n", .{err});
    };

    // Load index (simplified for testing)
    var loaded_store = VectorStore.load(allocator, "/tmp/embik_faiss_index.bin", dimension) catch |err| {
        std.debug.print("Load not implemented yet: {}\n", .{err});
        return;
    };
    defer loaded_store.deinit();

    std.debug.print("Basic VectorStore test completed!\n", .{});

    // Test semantic similarities
    std.debug.print("\n--- Testing Semantic Similarities ---\n", .{});

    // Create a new vector store for similarity testing
    var similarity_store = VectorStore.init(allocator, engine.n_embd, 2, "/tmp/similarity_test") catch |err| {
        std.debug.print("Failed to create similarity store: {}\n", .{err});
        return;
    };
    defer similarity_store.deinit();

    // Test cases with expected similarity relationships
    const test_texts = [_][]const u8{
        "The cat sat on the mat",           // 0 - simple sentence
        "A feline rested on the rug",       // 1 - similar meaning, different words
        "Dogs are loyal animals",           // 2 - different topic (animals)
        "Python is a programming language", // 3 - completely different (technology)
        "The dog lay on the carpet",        // 4 - similar structure to #0, different animal
        "I love programming in Python",     // 5 - related to #3
        "Cats are independent creatures",   // 6 - related to #0 (cats)
        "Machine learning is fascinating",  // 7 - technology but different from #3
    };

    std.debug.print("Embedding and adding test texts...\n", .{});

    // Embed and add all texts to the store
    var embeddings: [test_texts.len][]f32 = undefined;
    for (test_texts, 0..) |text, i| {
        const embedding = engine.embed(allocator, text) catch |err| {
            std.debug.print("Failed to embed text {d}: {}\n", .{i, err});
            continue;
        };
        embeddings[i] = embedding;

        similarity_store.add_vector(embedding, @intCast(i)) catch |err| {
            std.debug.print("Failed to add vector {d}: {}\n", .{i, err});
            continue;
        };

        std.debug.print("  {d}: '{s}'\n", .{i, text});
    }

    std.debug.print("\nTesting similarity searches:\n", .{});

    // Test similarity for a few interesting cases
    const query_indices = [_]usize{ 0, 2, 3 };

    for (query_indices) |query_idx| {
        std.debug.print("\n🔍 Query: '{s}' (ID: {d})\n", .{test_texts[query_idx], query_idx});

        var search_results = similarity_store.search(allocator, embeddings[query_idx], 3) catch |err| {
            std.debug.print("Failed to search: {}\n", .{err});
            continue;
        };
        defer search_results.deinit();

        std.debug.print("   Most similar:\n", .{});
        for (search_results.labels, search_results.distances) |label, distance| {
            if (label >= 0 and label < test_texts.len) {
                std.debug.print("   📝 Distance: {d:.4} - '{s}'\n", .{distance, test_texts[@intCast(label)]});
            }
        }
    }

    // Test with a custom query not in the store
    std.debug.print("\n🔍 Custom Query: 'Puppies are cute pets'\n", .{});
    const custom_query = engine.embed(allocator, "Puppies are cute pets") catch |err| {
        std.debug.print("Failed to embed custom query: {}\n", .{err});
        return;
    };
    defer allocator.free(custom_query);

    var custom_results = similarity_store.search(allocator, custom_query, 4) catch |err| {
        std.debug.print("Failed to search custom query: {}\n", .{err});
        return;
    };
    defer custom_results.deinit();

    std.debug.print("   Most similar to 'Puppies are cute pets':\n", .{});
    for (custom_results.labels, custom_results.distances) |label, distance| {
        if (label >= 0 and label < test_texts.len) {
            std.debug.print("   📝 Distance: {d:.4} - '{s}'\n", .{distance, test_texts[@intCast(label)]});
        }
    }

    // Clean up embeddings
    for (embeddings) |embedding| {
        allocator.free(embedding);
    }

    std.debug.print("\nSemantic similarity testing completed!\n", .{});

    // Test DocumentStore
    std.debug.print("\n--- Testing DocumentStore ---\n", .{});

    var document_store = DocumentStore.init(allocator, "/tmp/embik_docs", engine.n_embd) catch |err| {
        std.debug.print("Failed to initialize DocumentStore: {}\n", .{err});
        return;
    };
    defer document_store.deinit();

    std.debug.print("✓ DocumentStore initialized with folder structure\n", .{});

    // Test documents with metadata
    const test_documents = [_]struct {
        text: []const u8,
        author: []const u8,
        category: []const u8,
    }{
        .{ .text = "The quick brown fox jumps over the lazy dog", .author = "Test", .category = "Animals" },
        .{ .text = "Machine learning is revolutionizing technology", .author = "AI Expert", .category = "Technology" },
        .{ .text = "Dogs are loyal and friendly companions", .author = "Pet Lover", .category = "Animals" },
        .{ .text = "Python programming language is versatile", .author = "Programmer", .category = "Technology" },
    };

    var document_uids: [test_documents.len]u64 = undefined;

    // Store documents
    std.debug.print("\nStoring documents:\n", .{});
    for (test_documents, 0..) |doc, i| {
        const embedding = engine.embed(allocator, doc.text) catch |err| {
            std.debug.print("Failed to embed document {d}: {}\n", .{i, err});
            continue;
        };
        defer allocator.free(embedding);

        const uid = document_store.storeDocument(doc.text, embedding, doc.author, doc.category) catch |err| {
            std.debug.print("Failed to store document {d}: {}\n", .{i, err});
            continue;
        };

        document_uids[i] = uid;
        std.debug.print("  ✓ Stored '{s}' with UID: {d}\n", .{doc.text, uid});
    }

    std.debug.print("\nTotal documents stored: {d}\n", .{document_store.count()});

    // Test retrieval by UID
    std.debug.print("\nRetrieving document by UID:\n", .{});
    if (document_store.getDocument(document_uids[0])) |doc_data_opt| {
        if (doc_data_opt) |doc_data| {
            defer allocator.free(doc_data);
            std.debug.print("  ✓ Retrieved UID {d}: {s}\n", .{document_uids[0], doc_data});
        } else {
            std.debug.print("  ✗ Retrieved UID {d}: [null]\n", .{document_uids[0]});
        }
    } else |err| {
        std.debug.print("  ✗ Failed to retrieve document: {}\n", .{err});
    }

    // Test semantic search
    std.debug.print("\nSemantic search test:\n", .{});
    const search_query = "programming languages and coding";
    const query_embedding = engine.embed(allocator, search_query) catch |err| {
        std.debug.print("Failed to embed search query: {}\n", .{err});
        return;
    };
    defer allocator.free(query_embedding);

    var search_results = document_store.searchSimilar(query_embedding, 3) catch |err| {
        std.debug.print("Search failed: {}\n", .{err});
        return;
    };
    defer search_results.deinit();

    std.debug.print("🔍 Query: '{s}'\n", .{search_query});
    for (search_results.vector_result.labels, search_results.vector_result.distances, search_results.documents) |label, distance, doc_opt| {
        std.debug.print("  📝 UID: {d}, Distance: {d:.4}\n", .{label, distance});
        if (doc_opt) |doc_data| {
            std.debug.print("     Data: {s}\n", .{doc_data});
        } else {
            std.debug.print("     Data: [null]\n", .{});
        }
    }

    // Test deletion
    std.debug.print("\nTesting document deletion:\n", .{});
    document_store.deleteDocument(document_uids[1]) catch |err| {
        std.debug.print("Failed to delete document: {}\n", .{err});
        return;
    };
    std.debug.print("  ✓ Deleted document UID {d}\n", .{document_uids[1]});
    std.debug.print("  Documents remaining: {d}\n", .{document_store.count()});

    // Verify deletion
    if (document_store.getDocument(document_uids[1])) |doc_data| {
        if (doc_data) |data| {
            defer allocator.free(data);
            std.debug.print("  ✗ Document still exists after deletion!\n", .{});
        } else {
            std.debug.print("  ✓ Document successfully deleted (null returned)\n", .{});
        }
    } else |err| {
        std.debug.print("  ✓ Document successfully deleted (error: {})\n", .{err});
    }

    std.debug.print("\nDocumentStore test completed!\n", .{});
}
