//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const embed = @import("embed.zig");
pub const storage = @import("storage.zig");



// # Tests

// ## Embedding Engine Tests

test "embed init and deinit" {
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    try std.testing.expect(engine.model != null);
}

test "embed text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "Hello, world!";
    const embedding = try engine.embed(alloc, text);
    defer alloc.free(embedding);

    try std.testing.expect(embedding.len == engine.n_embd);
}

test "embed idempotency" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "The quick brown fox jumps over the lazy dog.";
    const embedding1 = try engine.embed(alloc, text);
    defer alloc.free(embedding1);

    const embedding2 = try engine.embed(alloc, text);
    defer alloc.free(embedding2);

    try std.testing.expect(embedding1.len == embedding2.len);
    for (embedding1, embedding2) |val1, val2| {
        try std.testing.expect(@abs(val1 - val2) < 0.0001);
    }
}

test "embed perf small text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "Hello, world!";
    const iterations = 100;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }
    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <5ms
    try std.testing.expect(@divFloor(duration, iterations) < 5);
}

test "embed perf large text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "The quick brown fox jumps over the lazy dog. ";
    const iterations = 20;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }

    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <8ms
    try std.testing.expect(@divFloor(duration, iterations) < 8);
}

test "embed perf extra large text" {
    const alloc = std.testing.allocator;
    const model_path = "deps/nomic-embed-text-v2-gguf/nomic-embed-text-v2-moe-q8_0.gguf";
    var engine = try embed.EmbeddingEngine.init(model_path);
    defer engine.deinit();

    const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
    const iterations = 10;

    const start = std.time.milliTimestamp();
    for (0..iterations) |_| {
        const embedding = try engine.embed(alloc, text);
        defer alloc.free(embedding);
    }

    const end = std.time.milliTimestamp();
    const duration = end - start;

    // expected operation cost <15ms
    try std.testing.expect(@divFloor(duration, iterations) < 15);
}


// ## Storage Engine Tests

test "MetadataStore init and deinit" {
    const alloc = std.testing.allocator;
    var metadata_store = try storage.MetadataStore.init(alloc, "test_metadata_db");
    defer metadata_store.deinit();
}

test "MetadataStore put and get" {
    const alloc = std.testing.allocator;
    var metadata_store = try storage.MetadataStore.init(alloc, "test_metadata_db_put_get");
    defer metadata_store.deinit();

    const key = "test_key";
    const value = "test_value";

    try metadata_store.put(key, value);

    const retrieved_value = try metadata_store.get(key);
    defer if (retrieved_value) |v| alloc.free(v);

    try std.testing.expect(retrieved_value != null);
    try std.testing.expectEqualStrings(value, retrieved_value.?);
}

test "MetadataStore get non-existent key" {
    const alloc = std.testing.allocator;
    var metadata_store = try storage.MetadataStore.init(alloc, "test_metadata_db_non_existent");
    defer metadata_store.deinit();

    const retrieved_value = try metadata_store.get("non_existent_key");
    try std.testing.expect(retrieved_value == null);
}

test "MetadataStore delete" {
    const alloc = std.testing.allocator;
    var metadata_store = try storage.MetadataStore.init(alloc, "test_metadata_db_delete");
    defer metadata_store.deinit();

    const key = "delete_test_key";
    const value = "delete_test_value";

    try metadata_store.put(key, value);

    var retrieved_value = try metadata_store.get(key);
    if (retrieved_value) |v| {
        alloc.free(v);
    }
    try std.testing.expect(retrieved_value != null);

    try metadata_store.delete(key);

    retrieved_value = try metadata_store.get(key);
    try std.testing.expect(retrieved_value == null);
}

test "VectorStore init and deinit" {
    const alloc = std.testing.allocator;
    var vector_store = try storage.VectorStore.init(alloc, 768, 2, "test_vector_store");
    defer vector_store.deinit();

    try std.testing.expect(vector_store.dimension == 768);
    try std.testing.expect(vector_store.ntotal() == 0);
}

test "VectorStore add_vector and search" {
    const alloc = std.testing.allocator;
    var vector_store = try storage.VectorStore.init(alloc, 3, 2, "test_vector_store_add");
    defer vector_store.deinit();

    const vector1 = [_]f32{ 1.0, 0.0, 0.0 };
    const vector2 = [_]f32{ 0.0, 1.0, 0.0 };
    const id1: i64 = 100;
    const id2: i64 = 200;

    try vector_store.add_vector(&vector1, id1);
    try vector_store.add_vector(&vector2, id2);

    try std.testing.expect(vector_store.ntotal() == 2);

    const query = [_]f32{ 1.0, 0.0, 0.0 };
    var search_result = try vector_store.search(alloc, &query, 2);
    defer search_result.deinit();

    try std.testing.expect(search_result.labels.len == 2);
    try std.testing.expect(search_result.distances.len == 2);
    try std.testing.expect(search_result.labels[0] == id1);
    try std.testing.expect(search_result.distances[0] < 0.1);
}

test "VectorStore delete_vector" {
    const alloc = std.testing.allocator;
    var vector_store = try storage.VectorStore.init(alloc, 3, 2, "test_vector_store_delete");
    defer vector_store.deinit();

    const vector1 = [_]f32{ 1.0, 0.0, 0.0 };
    const vector2 = [_]f32{ 0.0, 1.0, 0.0 };
    const id1: i64 = 300;
    const id2: i64 = 400;

    try vector_store.add_vector(&vector1, id1);
    try vector_store.add_vector(&vector2, id2);
    try std.testing.expect(vector_store.ntotal() == 2);

    try vector_store.delete_vector(id1);
    try std.testing.expect(vector_store.ntotal() == 1);

    const query = [_]f32{ 1.0, 0.0, 0.0 };
    var search_result = try vector_store.search(alloc, &query, 2);
    defer search_result.deinit();

    var found_id1 = false;
    for (search_result.labels) |label| {
        if (label == id1) found_id1 = true;
    }
    try std.testing.expect(!found_id1);
}

test "VectorStore dimension mismatch" {
    const alloc = std.testing.allocator;
    var vector_store = try storage.VectorStore.init(alloc, 3, 2, "test_vector_store_dimension");
    defer vector_store.deinit();

    const wrong_dimension_vector = [_]f32{ 1.0, 0.0 };
    const id: i64 = 500;

    const result = vector_store.add_vector(&wrong_dimension_vector, id);
    try std.testing.expectError(error.DimensionMismatch, result);
}

test "DocumentStore init and deinit" {
    const alloc = std.testing.allocator;
    var doc_store = try storage.DocumentStore.init(alloc, "test_doc_store", 3);
    defer doc_store.deinit();

    try std.testing.expect(doc_store.count() == 0);
}

test "DocumentStore storeDocument and getDocument" {
    const alloc = std.testing.allocator;
    var doc_store = try storage.DocumentStore.init(alloc, "test_doc_store_store_get", 3);
    defer doc_store.deinit();

    const text = "Hello, world!";
    const embedding = [_]f32{ 1.0, 0.0, 0.0 };
    const author = "test_author";
    const category = "test_category";

    const uid = try doc_store.storeDocument(text, &embedding, author, category);
    try std.testing.expect(uid > 0);
    try std.testing.expect(doc_store.count() == 1);

    const retrieved_doc = try doc_store.getDocument(uid);
    defer if (retrieved_doc) |doc| alloc.free(doc);

    try std.testing.expect(retrieved_doc != null);
    try std.testing.expect(std.mem.indexOf(u8, retrieved_doc.?, text) != null);
    try std.testing.expect(std.mem.indexOf(u8, retrieved_doc.?, author) != null);
    try std.testing.expect(std.mem.indexOf(u8, retrieved_doc.?, category) != null);
}

test "DocumentStore searchSimilar" {
    const alloc = std.testing.allocator;
    var doc_store = try storage.DocumentStore.init(alloc, "test_doc_store_search", 3);
    defer doc_store.deinit();

    const text1 = "First document";
    const embedding1 = [_]f32{ 1.0, 0.0, 0.0 };
    const text2 = "Second document";
    const embedding2 = [_]f32{ 0.0, 1.0, 0.0 };

    const uid1 = try doc_store.storeDocument(text1, &embedding1, "author1", "category1");
    const uid2 = try doc_store.storeDocument(text2, &embedding2, "author2", "category2");

    try std.testing.expect(doc_store.count() == 2);

    const query = [_]f32{ 1.0, 0.0, 0.0 };
    var search_results = try doc_store.searchSimilar(&query, 2);
    defer search_results.deinit();

    try std.testing.expect(search_results.vector_result.labels.len == 2);
    try std.testing.expect(search_results.documents.len == 2);

    const closest_uid = @as(u64, @intCast(search_results.vector_result.labels[0]));
    try std.testing.expect(closest_uid == uid1);
    try std.testing.expect(search_results.documents[0] != null);
    try std.testing.expect(std.mem.indexOf(u8, search_results.documents[0].?, text1) != null);

    _ = uid2;
}

test "DocumentStore deleteDocument" {
    const alloc = std.testing.allocator;
    var doc_store = try storage.DocumentStore.init(alloc, "test_doc_store_delete", 3);
    defer doc_store.deinit();

    const text = "Document to delete";
    const embedding = [_]f32{ 1.0, 0.0, 0.0 };
    const uid = try doc_store.storeDocument(text, &embedding, "author", "category");

    try std.testing.expect(doc_store.count() == 1);

    var retrieved_doc = try doc_store.getDocument(uid);
    if (retrieved_doc) |doc| {
        alloc.free(doc);
    }
    try std.testing.expect(retrieved_doc != null);

    try doc_store.deleteDocument(uid);
    try std.testing.expect(doc_store.count() == 0);

    retrieved_doc = try doc_store.getDocument(uid);
    try std.testing.expect(retrieved_doc == null);
}

test "DocumentStore unique IDs" {
    const alloc = std.testing.allocator;
    var doc_store = try storage.DocumentStore.init(alloc, "test_doc_store_unique_ids", 3);
    defer doc_store.deinit();

    const text = "Test document";
    const embedding = [_]f32{ 1.0, 0.0, 0.0 };

    const uid1 = try doc_store.storeDocument(text, &embedding, "author1", "category1");
    const uid2 = try doc_store.storeDocument(text, &embedding, "author2", "category2");
    const uid3 = try doc_store.storeDocument(text, &embedding, "author3", "category3");

    try std.testing.expect(uid1 != uid2);
    try std.testing.expect(uid2 != uid3);
    try std.testing.expect(uid1 != uid3);
    try std.testing.expect(doc_store.count() == 3);
}

test "MetadataStore perf put operations" {
    const alloc = std.testing.allocator;
    var metadata_store = try storage.MetadataStore.init(alloc, "test_metadata_perf_put");
    defer metadata_store.deinit();

    const iterations = 1000;
    const start = std.time.microTimestamp();

    for (0..iterations) |i| {
        const key = try std.fmt.allocPrint(alloc, "key_{d}", .{i});
        defer alloc.free(key);
        const value = try std.fmt.allocPrint(alloc, "value_{d}", .{i});
        defer alloc.free(value);

        try metadata_store.put(key, value);
    }

    const end = std.time.microTimestamp();
    const duration = end - start;

    // Expected < 60μs per operation (allowing for system variance)
    try std.testing.expect(@divFloor(duration, iterations) < 60);
}

test "MetadataStore perf get operations" {
    const alloc = std.testing.allocator;
    var metadata_store = try storage.MetadataStore.init(alloc, "test_metadata_perf_get");
    defer metadata_store.deinit();

    // Pre-populate with test data
    const iterations = 1000;
    for (0..iterations) |i| {
        const key = try std.fmt.allocPrint(alloc, "key_{d}", .{i});
        defer alloc.free(key);
        const value = try std.fmt.allocPrint(alloc, "value_{d}", .{i});
        defer alloc.free(value);

        try metadata_store.put(key, value);
    }

    const start = std.time.microTimestamp();

    for (0..iterations) |i| {
        const key = try std.fmt.allocPrint(alloc, "key_{d}", .{i});
        defer alloc.free(key);

        const retrieved = try metadata_store.get(key);
        if (retrieved) |val| alloc.free(val);
    }

    const end = std.time.microTimestamp();
    const duration = end - start;

    // Expected < 30μs per operation (rounded up from 24μs)
    try std.testing.expect(@divFloor(duration, iterations) < 30);
}

test "VectorStore perf add and search" {
    const alloc = std.testing.allocator;
    var vector_store = try storage.VectorStore.init(alloc, 128, 2, "test_vector_perf");
    defer vector_store.deinit();

    // Add vectors performance test
    const add_iterations = 500;
    const vectors = try alloc.alloc([128]f32, add_iterations);
    defer alloc.free(vectors);

    // Initialize test vectors
    for (vectors, 0..) |*vec, i| {
        for (vec, 0..) |*val, j| {
            val.* = @as(f32, @floatFromInt(i + j)) / 1000.0;
        }
    }

    const add_start = std.time.microTimestamp();

    for (vectors, 0..) |*vec, i| {
        try vector_store.add_vector(vec, @as(i64, @intCast(i)));
    }

    const add_end = std.time.microTimestamp();
    const add_duration = add_end - add_start;

    // Expected < 10μs per operation (rounded up from 0μs, allowing for variance)
    try std.testing.expect(@divFloor(add_duration, add_iterations) < 10);

    // Search performance test
    const search_iterations = 100;
    const query_vec = vectors[0];

    const search_start = std.time.microTimestamp();

    for (0..search_iterations) |_| {
        var search_result = try vector_store.search(alloc, &query_vec, 10);
        defer search_result.deinit();
    }

    const search_end = std.time.microTimestamp();
    const search_duration = search_end - search_start;

    // Expected < 100μs per operation (rounded up from 95μs)
    try std.testing.expect(@divFloor(search_duration, search_iterations) < 100);
}

test "DocumentStore perf end-to-end operations" {
    const alloc = std.testing.allocator;
    var doc_store = try storage.DocumentStore.init(alloc, "test_doc_store_perf", 64);
    defer doc_store.deinit();

    // Store documents performance test
    const store_iterations = 200;
    const embeddings = try alloc.alloc([64]f32, store_iterations);
    defer alloc.free(embeddings);

    // Initialize test embeddings
    for (embeddings, 0..) |*emb, i| {
        for (emb, 0..) |*val, j| {
            val.* = @as(f32, @floatFromInt(i * j)) / 10000.0;
        }
    }

    const store_start = std.time.microTimestamp();
    var uids = try alloc.alloc(u64, store_iterations);
    defer alloc.free(uids);

    for (embeddings, 0..) |*emb, i| {
        const text = try std.fmt.allocPrint(alloc, "Document text {d}", .{i});
        defer alloc.free(text);
        const author = try std.fmt.allocPrint(alloc, "author_{d}", .{i});
        defer alloc.free(author);

        uids[i] = try doc_store.storeDocument(text, emb, author, "test_category");
    }

    const store_end = std.time.microTimestamp();
    const store_duration = store_end - store_start;

    // Expected < 150μs per operation (allowing for system variance)
    try std.testing.expect(@divFloor(store_duration, store_iterations) < 150);

    // Search performance test
    const search_iterations = 50;
    const query_emb = embeddings[0];

    const search_start = std.time.microTimestamp();

    for (0..search_iterations) |_| {
        var search_results = try doc_store.searchSimilar(&query_emb, 5);
        defer search_results.deinit();
    }

    const search_end = std.time.microTimestamp();
    const search_duration = search_end - search_start;

    // Expected < 320μs per operation (rounded up from 309μs)
    try std.testing.expect(@divFloor(search_duration, search_iterations) < 320);
}
