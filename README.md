# embik

A high-performance CLI tool for intelligent file indexing and semantic search using vector embeddings.

## Vision

embik enables developers to create searchable knowledge bases from their codebases and documents. By combining tree-sitter for intelligent code chunking, nomic embeddings via llama.cpp, FAISS for vector storage, and RocksDB for metadata persistence, embik provides fast, semantic search capabilities across large file collections.

## Architecture

- **Tree-sitter**: Intelligent chunking of source code respecting language semantics
- **Nomic + llama.cpp**: High-quality text embeddings using the nomic-embed-text-v2 model
- **FAISS**: Efficient vector similarity search and storage
- **RocksDB**: Persistent metadata storage for file information and chunk mappings
- **Zig**: Systems-level performance with memory safety

## Current Status

**Working embedding engine and metadata storage** - the project has functional embedding system that can load the nomic-embed-text-v2 model and generate 768-dimensional embeddings for text inputs, plus a complete RocksDB-based metadata storage layer.

### Implemented ✓
- **EmbeddingEngine** (`src/embed.zig`) - Complete llama.cpp integration for text embeddings
  - Model loading and context management with error handling
  - Text tokenization using llama.cpp vocabulary
  - Batch processing for efficient embedding generation
  - Mean pooling for sequence-level embeddings
  - Memory-safe resource cleanup and management
- **MetadataStore** (`src/storage.zig`) - RocksDB integration for persistent metadata storage
  - Database initialization with proper error handling
  - Key-value operations: put, get, delete
  - Memory management for retrieved values
  - Thread-safe database operations
- **Demo application** (`src/main.zig`) - Working example that embeds sample texts and tests storage
- **Build system** (`build.zig`) - Proper dependency management linking faiss, llama.cpp, and rocksdb
- **Module structure** (`src/root.zig`) - Clean library organization with public API exports

### Architecture Files
- `src/embed.zig` - Core embedding functionality (101 lines, fully implemented)
- `src/main.zig` - Demo application showing embedding and storage workflow (104 lines, functional)
- `src/root.zig` - Module exports and library interface (5 lines, complete)
- `src/storage.zig` - RocksDB metadata storage implementation (121 lines, fully implemented)
- `build.zig` - Build configuration with C library linking (71 lines, complete)

### Current Demo Capabilities
The current implementation can:
- Load the nomic-embed-text-v2 model from the local `deps/` directory
- Tokenize input text using the model's vocabulary
- Generate 768-dimensional embeddings for any text input
- Process multiple texts in sequence with proper memory management
- Store and retrieve metadata using RocksDB with full CRUD operations
- Display embedding vectors and storage test results for debugging

### Next Development Phase

#### Core CLI Interface
- [ ] Command-line argument parsing with subcommands
- [ ] File indexing commands (`embik index <path>`)
- [ ] Search commands (`embik search <query>`)
- [ ] Configuration management and user settings

#### File Processing Pipeline
- [ ] Tree-sitter integration for language-aware code parsing
- [ ] Intelligent chunking strategies per file type
- [ ] File type detection and handling
- [ ] Recursive directory traversal with filtering
- [ ] File change detection and incremental indexing

#### Vector Storage & Search
- [ ] FAISS index creation and management
- [ ] Efficient vector similarity search implementation
- [ ] Index persistence and loading from disk
- [ ] Search result ranking and filtering
- [ ] Similarity threshold configuration

#### Metadata Management
- [x] RocksDB integration for metadata storage
- [ ] File metadata persistence (path, size, modified time, etc.)
- [ ] Chunk-to-file mapping and reverse lookups
- [ ] Index statistics and management commands

#### Advanced Features
- [ ] Multiple embedding models support
- [ ] Custom chunking strategies per file type
- [ ] Search result context expansion
- [ ] Index optimization and compression
- [ ] Configuration profiles for different use cases

## Installation

### Dependencies
```bash
# macOS (Homebrew)
brew install faiss llama.cpp rocksdb zig
```

### Build
```bash
zig build
```

### Run
```bash
# Current demo - embeds sample texts and tests metadata storage
zig build run

# Future CLI usage (planned)
./zig-out/bin/embik index ~/code
./zig-out/bin/embik search "error handling patterns"
```

## Model Setup

The project includes the nomic-embed-text-v2 model in `deps/nomic-embed-text-v2-gguf/`. This provides 768-dimensional embeddings optimized for semantic similarity tasks.

## Development

Built with Zig 0.15.1+, leveraging system libraries for maximum performance while maintaining memory safety.

Current development focus: Transitioning from proof-of-concept embedding engine to building out the CLI interface and file processing pipeline for a complete semantic search tool.

### Project Structure
```
embik/
├── src/
│   ├── embed.zig      # Core embedding functionality (complete)
│   ├── main.zig       # Demo application (functional)
│   ├── root.zig       # Module exports (complete)
│   └── storage.zig    # Future storage layer (placeholder)
├── deps/
│   └── nomic-embed-text-v2-gguf/  # Pre-trained model
├── build.zig          # Build configuration
└── README.md          # This file
```