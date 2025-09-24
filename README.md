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

**Basic embedding functionality is working** - the project can currently load the nomic embedding model and generate embeddings for text inputs.

### Implemented 
- Embedding engine with nomic-embed-text-v2 model integration
- llama.cpp bindings for text embedding
- Basic tokenization and embedding generation
- Build system with dependency management (faiss, llama.cpp, rocksdb)

### In Progress / Planned =§

#### Core CLI Interface
- [ ] Command-line argument parsing
- [ ] File indexing commands (`embik index <path>`)
- [ ] Search commands (`embik search <query>`)
- [ ] Configuration management

#### File Processing Pipeline
- [ ] Tree-sitter integration for code parsing
- [ ] Language-aware chunking strategies
- [ ] File type detection and handling
- [ ] Recursive directory traversal
- [ ] File change detection and incremental indexing

#### Vector Storage & Search
- [ ] FAISS index creation and management
- [ ] Efficient vector similarity search
- [ ] Index persistence and loading
- [ ] Search result ranking and filtering

#### Metadata Management
- [ ] RocksDB integration for metadata storage
- [ ] File metadata persistence (path, size, modified time, etc.)
- [ ] Chunk-to-file mapping
- [ ] Index statistics and management

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
# Current demo
zig build run

# Future CLI usage
./zig-out/bin/embik index ~/code
./zig-out/bin/embik search "error handling patterns"
```

## Model Setup

The project includes the nomic-embed-text-v2 model in `deps/nomic-embed-text-v2-gguf/`. This provides 768-dimensional embeddings optimized for semantic similarity.

## Development

Built with Zig 0.15.1+, leveraging system libraries for maximum performance while maintaining memory safety.

Current development focus: Building out the CLI interface and file processing pipeline to move from proof-of-concept to functional tool.