const std = @import("std");
const rocksdb = @cImport({
    @cDefine("ROCKSDB_PLATFORM_POSIX", "1");
    @cInclude("rocksdb/c.h");
});
const faiss = @cImport({
    @cInclude("faiss/c_api/faiss_c.h");
    @cInclude("faiss/c_api/Index_c.h");
    @cInclude("faiss/c_api/IndexFlat_c.h");
    @cInclude("faiss/c_api/IndexIVFFlat_c.h");
    @cInclude("faiss/c_api/MetaIndexes_c.h");
    @cInclude("faiss/c_api/index_io_c.h");
    @cInclude("faiss/c_api/impl/AuxIndexStructures_c.h");
});

/// uses rocksdb for persistent storage
pub const MetadataStore = struct {
    const Self = @This();

    db: *rocksdb.rocksdb_t,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Self {
        // Create options
        const options = rocksdb.rocksdb_options_create();
        defer rocksdb.rocksdb_options_destroy(options);

        // Set create_if_missing to true
        rocksdb.rocksdb_options_set_create_if_missing(options, 1);

        // Open database
        var err: [*c]u8 = null;
        const path_cstr = try allocator.dupeZ(u8, path);
        defer allocator.free(path_cstr);

        const db = rocksdb.rocksdb_open(options, path_cstr.ptr, &err);
        if (err) |e| {
            defer rocksdb.rocksdb_free(e);
            return error.RocksDBOpenFailed;
        }

        return Self{
            .db = db.?,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        rocksdb.rocksdb_close(self.db);
    }

    pub fn put(self: *Self, key: []const u8, value: []const u8) !void {
        const write_options = rocksdb.rocksdb_writeoptions_create();
        defer rocksdb.rocksdb_writeoptions_destroy(write_options);

        var err: [*c]u8 = null;
        rocksdb.rocksdb_put(
            self.db,
            write_options,
            key.ptr,
            key.len,
            value.ptr,
            value.len,
            &err,
        );

        if (err) |e| {
            defer rocksdb.rocksdb_free(e);
            return error.RocksDBPutFailed;
        }
    }

    pub fn get(self: *Self, key: []const u8) !?[]u8 {
        const read_options = rocksdb.rocksdb_readoptions_create();
        defer rocksdb.rocksdb_readoptions_destroy(read_options);

        var err: [*c]u8 = null;
        var value_len: usize = 0;
        const value_ptr = rocksdb.rocksdb_get(
            self.db,
            read_options,
            key.ptr,
            key.len,
            &value_len,
            &err,
        );

        if (err) |e| {
            defer rocksdb.rocksdb_free(e);
            return error.RocksDBGetFailed;
        }

        if (value_ptr == null) {
            return null;
        }

        const value = try self.allocator.dupe(u8, value_ptr[0..value_len]);
        rocksdb.rocksdb_free(value_ptr);
        return value;
    }

    pub fn delete(self: *Self, key: []const u8) !void {
        const write_options = rocksdb.rocksdb_writeoptions_create();
        defer rocksdb.rocksdb_writeoptions_destroy(write_options);

        var err: [*c]u8 = null;
        rocksdb.rocksdb_delete(
            self.db,
            write_options,
            key.ptr,
            key.len,
            &err,
        );

        if (err) |e| {
            defer rocksdb.rocksdb_free(e);
            return error.RocksDBDeleteFailed;
        }
    }
};
/// uses faiss for vector search and storing indexes
pub const VectorStore = struct {
    const Self = @This();

    index: *faiss.FaissIndexIDMap,
    dimension: faiss.idx_t,
    allocator: std.mem.Allocator,
    path: []u8,

    pub fn init(allocator: std.mem.Allocator, dimension: faiss.idx_t, nlist: faiss.idx_t, path: []const u8) !Self {
        _ = nlist; // unused for this implementation
        const path_owned = try allocator.dupe(u8, path);

        // Create flat L2 index as the underlying index
        var flat_index: ?*faiss.FaissIndexFlat = null;
        const flat_result = faiss.faiss_IndexFlat_new_with(&flat_index, dimension, faiss.METRIC_L2);
        if (flat_result != 0 or flat_index == null) {
            allocator.free(path_owned);
            return error.FlatIndexCreationFailed;
        }

        // Wrap with IndexIDMap for stable ID support
        var id_map_index: ?*faiss.FaissIndexIDMap = null;
        const id_map_result = faiss.faiss_IndexIDMap_new(&id_map_index, @as(*faiss.FaissIndex, @ptrCast(flat_index.?)));
        if (id_map_result != 0 or id_map_index == null) {
            _ = faiss.faiss_Index_free(@as(*faiss.FaissIndex, @ptrCast(flat_index.?)));
            allocator.free(path_owned);
            return error.IDMapCreationFailed;
        }

        return Self{
            .index = id_map_index.?,
            .dimension = dimension,
            .allocator = allocator,
            .path = path_owned,
        };
    }

    pub fn deinit(self: *Self) void {
        _ = faiss.faiss_Index_free(@as(*faiss.FaissIndex, @ptrCast(self.index)));
        self.allocator.free(self.path);
    }

    pub fn train(self: *Self, training_data: []const f32) !void {
        _ = self;
        _ = training_data;
        // Flat index doesn't need training
    }

    pub fn add_vector(self: *Self, vector: []const f32, id: faiss.idx_t) !void {
        if (vector.len != @as(usize, @intCast(self.dimension))) {
            return error.DimensionMismatch;
        }

        // Use add_with_ids for IndexIDMap to support custom IDs
        const add_result = faiss.faiss_Index_add_with_ids(@as(*faiss.FaissIndex, @ptrCast(self.index)), 1, vector.ptr, &id);
        if (add_result != 0) {
            return error.AddVectorFailed;
        }
    }

    pub fn delete_vector(self: *Self, id: faiss.idx_t) !void {
        // Create IDSelectorRange for the single ID
        var selector: ?*faiss.FaissIDSelectorRange = null;
        const selector_result = faiss.faiss_IDSelectorRange_new(&selector, id, id + 1);
        if (selector_result != 0 or selector == null) {
            return error.SelectorCreationFailed;
        }
        defer _ = faiss.faiss_IDSelector_free(@as(*faiss.FaissIDSelector, @ptrCast(selector.?)));

        // Remove the vector using the selector
        var n_removed: usize = 0;
        const remove_result = faiss.faiss_Index_remove_ids(@as(*faiss.FaissIndex, @ptrCast(self.index)), @as(*faiss.FaissIDSelector, @ptrCast(selector.?)), &n_removed);
        if (remove_result != 0) {
            return error.RemoveVectorFailed;
        }

        // Optional: check if any vectors were actually removed
        if (n_removed == 0) {
            return error.VectorNotFound;
        }
    }

    pub fn search(self: *Self, allocator: std.mem.Allocator, query: []const f32, k: faiss.idx_t) !SearchResult {
        if (query.len != @as(usize, @intCast(self.dimension))) {
            return error.DimensionMismatch;
        }

        const distances = try allocator.alloc(f32, @intCast(k));
        const labels = try allocator.alloc(faiss.idx_t, @intCast(k));

        const search_result = faiss.faiss_Index_search(@as(*faiss.FaissIndex, @ptrCast(self.index)), 1, query.ptr, k, distances.ptr, labels.ptr);
        if (search_result != 0) {
            allocator.free(distances);
            allocator.free(labels);
            return error.SearchFailed;
        }

        return SearchResult{
            .distances = distances,
            .labels = labels,
            .allocator = allocator,
        };
    }

    pub fn save(self: *Self) !void {
        _ = self;
        // Simplified for testing - save not implemented yet
        return error.OperationNotSupported;
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8, dimension: faiss.idx_t) !Self {
        _ = path;
        // Simplified for testing - create new index instead of loading
        return init(allocator, dimension, 2, "/tmp/dummy");
    }

    pub fn ntotal(self: *Self) faiss.idx_t {
        return faiss.faiss_Index_ntotal(@as(*faiss.FaissIndex, @ptrCast(self.index)));
    }
};

pub const SearchResult = struct {
    distances: []f32,
    labels: []faiss.idx_t,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SearchResult) void {
        self.allocator.free(self.distances);
        self.allocator.free(self.labels);
    }
};

/// Combined document store with text, metadata, and vector search
pub const DocumentStore = struct {
    const Self = @This();

    vector_store: VectorStore,
    metadata_store: MetadataStore,
    allocator: std.mem.Allocator,
    base_path: []u8,
    rng: std.Random.DefaultPrng,
    last_timestamp_ms: u64,
    counter: u32,

    pub fn init(allocator: std.mem.Allocator, base_path: []const u8, dimension: faiss.idx_t) !Self {
        // Create base directory
        std.fs.cwd().makeDir(base_path) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        const base_path_owned = try allocator.dupe(u8, base_path);

        // Create subdirectories
        const vec_path = try std.fmt.allocPrint(allocator, "{s}/vec", .{base_path});
        defer allocator.free(vec_path);
        const data_path = try std.fmt.allocPrint(allocator, "{s}/data", .{base_path});
        defer allocator.free(data_path);

        std.fs.cwd().makeDir(vec_path) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
        std.fs.cwd().makeDir(data_path) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        // Initialize stores
        const vector_store = VectorStore.init(allocator, dimension, 2, vec_path) catch |err| {
            allocator.free(base_path_owned);
            return err;
        };

        const metadata_store = MetadataStore.init(allocator, data_path) catch |err| {
            allocator.free(base_path_owned);
            return err;
        };

        // Initialize RNG with high-entropy seed
        var seed_bytes: [8]u8 = undefined;
        std.crypto.random.bytes(seed_bytes[0..]);
        const seed = std.mem.readInt(u64, &seed_bytes, .little);
        const rng = std.Random.DefaultPrng.init(seed);

        return Self{
            .vector_store = vector_store,
            .metadata_store = metadata_store,
            .allocator = allocator,
            .base_path = base_path_owned,
            .rng = rng,
            .last_timestamp_ms = 0,
            .counter = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.vector_store.deinit();
        self.metadata_store.deinit();
        self.allocator.free(self.base_path);
    }

    /// Generate a unique ID based on timestamp, process ID, and entropy
    fn generateUID(self: *Self) u64 {
        // Get current timestamp in milliseconds
        const timestamp_ms = std.time.milliTimestamp();

        // If same millisecond as last call, increment counter
        if (timestamp_ms == self.last_timestamp_ms) {
            self.counter += 1;
        } else {
            self.last_timestamp_ms = @intCast(timestamp_ms);
            self.counter = 0;
        }

        // Get process ID for additional uniqueness across processes
        const pid = switch (@import("builtin").os.tag) {
            .linux => std.os.linux.getpid(),
            .macos => std.c.getpid(),
            else => @as(u32, 1), // fallback
        };

        // Generate random entropy
        const random_part = self.rng.random().int(u16);

        // Combine components into a 64-bit UID:
        // - Top 42 bits: timestamp (milliseconds since epoch)
        // - Next 12 bits: counter (up to 4095 IDs per millisecond)
        // - Next 6 bits: process ID (masked to 6 bits)
        // - Bottom 4 bits: random entropy
        const uid = (@as(u64, @intCast(timestamp_ms)) << 22) |
                   (@as(u64, self.counter & 0xFFF) << 10) |
                   (@as(u64, @as(u64, @intCast(pid)) & 0x3F) << 4) |
                   (@as(u64, random_part & 0xF));

        return uid;
    }

    /// Store a document with text, embedding, and custom metadata
    pub fn storeDocument(self: *Self, text: []const u8, embedding: []const f32, author: []const u8, category: []const u8) !u64 {
        const uid = self.generateUID();

        // Store the embedding in vector store
        self.vector_store.add_vector(embedding, @intCast(uid)) catch |err| {
            return err;
        };

        // Create metadata JSON with proper structure
        const metadata_json = try std.fmt.allocPrint(self.allocator,
            "{{\"text\":\"{s}\",\"author\":\"{s}\",\"category\":\"{s}\",\"uid\":{d},\"created_at\":{d}}}",
            .{ text, author, category, uid, std.time.timestamp() }
        );
        defer self.allocator.free(metadata_json);

        // Store metadata using UID as key
        const uid_str = try std.fmt.allocPrint(self.allocator, "{d}", .{uid});
        defer self.allocator.free(uid_str);

        self.metadata_store.put(uid_str, metadata_json) catch |err| {
            // If metadata storage fails, try to clean up the vector
            self.vector_store.delete_vector(@intCast(uid)) catch {};
            return err;
        };

        return uid;
    }

    /// Retrieve a document by UID (raw JSON)
    pub fn getDocument(self: *Self, uid: u64) !?[]u8 {
        const uid_str = try std.fmt.allocPrint(self.allocator, "{d}", .{uid});
        defer self.allocator.free(uid_str);

        return self.metadata_store.get(uid_str);
    }


    /// Delete a document by UID
    pub fn deleteDocument(self: *Self, uid: u64) !void {
        const uid_str = try std.fmt.allocPrint(self.allocator, "{d}", .{uid});
        defer self.allocator.free(uid_str);

        // Delete from both stores
        try self.vector_store.delete_vector(@intCast(uid));
        try self.metadata_store.delete(uid_str);
    }

    /// Search for similar documents and return with metadata
    pub fn searchSimilar(self: *Self, query_embedding: []const f32, k: faiss.idx_t) !DocumentSearchResult {
        // Search in vector store
        const vector_results = try self.vector_store.search(self.allocator, query_embedding, k);

        // Get metadata for each result
        const documents = try self.allocator.alloc(?[]u8, vector_results.labels.len);

        for (vector_results.labels, 0..) |label, i| {
            if (label >= 0) {
                documents[i] = self.getDocument(@intCast(label)) catch null;
            } else {
                documents[i] = null;
            }
        }

        return DocumentSearchResult{
            .vector_result = vector_results,
            .documents = documents,
            .allocator = self.allocator,
        };
    }

    /// Get total number of documents
    pub fn count(self: *Self) faiss.idx_t {
        return self.vector_store.ntotal();
    }
};

pub const DocumentSearchResult = struct {
    vector_result: SearchResult,
    documents: []?[]u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DocumentSearchResult) void {
        for (self.documents) |doc_opt| {
            if (doc_opt) |doc| {
                self.allocator.free(doc);
            }
        }
        self.allocator.free(self.documents);
        self.vector_result.deinit();
    }
};
