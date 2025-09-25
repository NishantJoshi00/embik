# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

embik is a high-performance document storage and semantic search engine written in Zig. It combines vector embeddings (nomic-embed-text-v2), FAISS for vector similarity search, and RocksDB for metadata persistence to create an intelligent document store with enterprise-grade UID generation.

## Architecture

### Core Components
- **EmbeddingEngine** (`src/embed.zig`) - llama.cpp integration for generating 768-dimensional embeddings
- **VectorStore** (`src/storage.zig`) - FAISS IndexIDMap wrapper for stable vector similarity search
- **MetadataStore** (`src/storage.zig`) - RocksDB integration for persistent JSON metadata storage
- **DocumentStore** (`src/storage.zig`) - Main unified API combining embedding, vector, and metadata operations

### Key Architecture Patterns
- **Entropy-based UID Generation**: 64-bit UIDs using timestamp (42 bits) + counter (12 bits) + process ID (6 bits) + entropy (4 bits)
- **Organized Storage Structure**: Separate `/vec` (FAISS) and `/data` (RocksDB) folders
- **C Library Integration**: Direct C bindings to FAISS, RocksDB, and llama.cpp for performance
- **Memory Safety**: Zig's safety guarantees with proper resource cleanup using `defer`

## Development Commands

### Build & Run
```bash
# Build the project
zig build

# Run the demo/test suite (shows all functionality)
zig build run

# Run tests
zig build test
```

### Dependencies
System dependencies are managed via Homebrew on macOS:
```bash
brew install faiss llama.cpp rocksdb zig
```

The embedding model should be placed in `deps/nomic-embed-text-v2-gguf/`.

## Code Structure

### Module Exports (`src/root.zig`)
- `embed` - EmbeddingEngine functionality
- `storage` - All storage components (DocumentStore, VectorStore, MetadataStore)

### Main Storage API (`src/storage.zig`)
The DocumentStore is the primary interface:
```zig
// Initialize with path and embedding dimension
var doc_store = try DocumentStore.init(allocator, "/path/to/store", 768);
defer doc_store.deinit();

// Store document with metadata
const uid = try doc_store.storeDocument(text, embedding, "author", "category");

// Semantic search
const results = try doc_store.searchSimilar(query_embedding, 5);
defer results.deinit();
```

### Build Configuration (`build.zig`)
- Links to system libraries: faiss_c, llama, rocksdb
- Uses Homebrew paths: `/opt/homebrew/include` and `/opt/homebrew/lib`
- Creates both library module and executable

## Development Guidelines

### Memory Management
- All allocations use the provided allocator
- Use `defer` for cleanup of resources (databases, search results, etc.)
- C library resources require explicit cleanup calls

### Error Handling
- Custom error types for each component (e.g., `error.ModelLoadFailed`, `error.RocksDBOpenFailed`)
- Proper error propagation using Zig's `!` syntax
- Resource cleanup on error paths

### ID Management
- UIDs are 64-bit integers generated with entropy-based algorithm
- FAISS IndexIDMap ensures stable ID-to-vector mapping
- Cross-platform UID generation with fallbacks

### Testing
The demo in `src/main.zig` serves as comprehensive integration tests covering:
- Embedding generation
- Storage operations (RocksDB)
- Vector similarity search (FAISS)
- Complete DocumentStore lifecycle
- Semantic search with metadata retrieval

## File Locations

Key files and their purposes:
- `src/embed.zig:18-50` - EmbeddingEngine initialization and model loading
- `src/storage.zig:23-46` - MetadataStore RocksDB setup
- `src/storage.zig:200-300` - DocumentStore main API implementation
- `build.zig:62-70` - System dependency linking configuration

## Development Notes

This is a complete implementation focused on core functionality. The architecture is production-ready with enterprise features like entropy-based UIDs, structured JSON metadata, and organized storage separation. Future development may add CLI interfaces and batch processing capabilities.