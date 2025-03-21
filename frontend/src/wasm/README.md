# PrimeCompress WebAssembly Integration

This directory contains files for integrating the PrimeCompress library with WebAssembly to enable in-browser compression.

## Current Implementation

The current implementation includes:

1. **Mock WebAssembly Interface** (`prime-compress-wasm.ts`)
   - Provides a TypeScript interface that matches the expected behavior of the final WebAssembly module
   - Includes mock implementations of compression and decompression functions
   - Implements a singleton pattern for managing the WebAssembly module

2. **Web Worker Integration** (`compression.worker.ts`)
   - Enables non-blocking compression operations using a Web Worker
   - Handles compression and decompression requests asynchronously
   - Provides consistent message passing interface

3. **Worker Manager Utility** (`/utils/worker-manager.ts`)
   - Manages communication with the Web Worker using a Promise-based API
   - Handles initialization, message passing, and worker lifecycle

## Pending Implementation

The following items still need to be implemented:

1. **Actual WebAssembly Build**
   - Compile the PrimeCompress library to WebAssembly using Emscripten
   - Create the necessary bindings to expose the core functionality
   - Replace the mock implementations with actual WebAssembly calls

2. **WebAssembly Integration**
   - Update the `prime-compress-wasm.ts` file to load and use the actual WebAssembly module
   - Implement proper memory management between JavaScript and WebAssembly
   - Handle different browser capabilities and fallback mechanisms

3. **Testing with Real Files**
   - Implement comprehensive testing for the WebAssembly module
   - Verify performance across different file types and sizes
   - Ensure compatibility across different browsers

## Building the WebAssembly Module

To build the WebAssembly module (future implementation):

1. Install Emscripten SDK following [official instructions](https://emscripten.org/docs/getting_started/downloads.html)
2. Create a build script to compile the PrimeCompress library to WebAssembly
3. Use the following command (example):

```bash
emcc -O3 \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS="['_compress', '_decompress', '_getAvailableStrategies']" \
  -s EXPORTED_RUNTIME_METHODS="['cwrap', 'setValue', 'getValue']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -o prime-compress.js \
  src/core/unified-compression.js src/core/compression-wrapper.js [other source files]
```

4. Place the compiled `.wasm` and `.js` files in an appropriate location in the `public` directory

## Usage in the Application

The WebAssembly module is used in the application through:

1. The `compression.ts` service, which provides a high-level API for compression operations
2. The `useCompression` hook, which exposes compression functionality to React components
3. The `FileUploadComponent`, which uses the hook to compress files
4. The `TestPage` component, which uses the service to test different compression strategies

## Debugging WebAssembly

For debugging WebAssembly issues:

1. Use the browser's developer tools to inspect memory and call stacks
2. Enable debug information when building WebAssembly with `-g4` flag
3. Log state transitions in the WebAssembly interface
4. Use the network tab to ensure WebAssembly files are loading correctly