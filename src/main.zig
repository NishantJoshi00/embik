const std = @import("std");
const embik = @import("embik");

const EmbeddingEngine = embik.embed.EmbeddingEngine;

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
}
