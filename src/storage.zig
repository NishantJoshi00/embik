const std = @import("std");
const rocksdb = @cImport({
    @cDefine("ROCKSDB_PLATFORM_POSIX", "1");
    @cInclude("rocksdb/c.h");
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

    pub fn init(path: []const u8) !Self {
        _ = path;
        // Initialize FAISS here
        return Self{};
    }
};
