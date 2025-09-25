

/// uses rocksdb for persistent storage
pub const MetadataStore = struct {



    const Self = @This();

    pub fn init(path: []const u8) !Self {
        // Initialize RocksDB here
        return Self{};
    }
};
/// uses faiss for vector search and storing indexes
pub const VectorStore = struct {

    const Self = @This();

    pub fn init(path: []const u8) !Self {
        // Initialize FAISS here
        return Self{};
    }
};
