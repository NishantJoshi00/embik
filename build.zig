const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("embik", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    linkDeps(mod);

    const exe = b.addExecutable(.{
        .name = "embik",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),

            .target = target,
            .optimize = optimize,

            .imports = &.{
                .{ .name = "embik", .module = mod },
            },
        }),
    });

    exe.linkLibC();
    exe.linkLibCpp();

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}

fn linkDeps(exe: *std.Build.Module) void {
    // Use Homebrew installation
    exe.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    exe.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });

    exe.linkSystemLibrary("faiss", .{ .needed = true }); // brew install faiss
    exe.linkSystemLibrary("llama", .{ .needed = true }); // brew install llama.cpp
    exe.linkSystemLibrary("rocksdb", .{ .needed = true }); // brew install rocksdb
}
