const std = @import("std");
const embik = @import("embik");

const EmbeddingEngine = embik.embed.EmbeddingEngine;
const MetadataStore = embik.storage.MetadataStore;

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
}
