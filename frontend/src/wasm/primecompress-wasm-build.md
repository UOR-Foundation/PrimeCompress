# PrimeCompress WebAssembly Build Guide

This document outlines the process for building the WebAssembly module for PrimeCompress.

## Prerequisites

1. Install Emscripten SDK:
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. Create a build directory:
   ```bash
   mkdir -p wasm-build
   cd wasm-build
   ```

## Building Process

### 1. Prepare the JavaScript files for compilation

Create an entry point file `primecompress-wasm.js`:

```javascript
// Include the necessary modules
const compression = require('../src/core/unified-compression.js');

// Export the functions that will be exposed to WebAssembly
module.exports = {
  compress: function(data, options) {
    // Convert options from JSON string to object
    const parsedOptions = options ? JSON.parse(options) : {};
    return compression.compress(data, parsedOptions);
  },

  decompress: function(compressedData) {
    return compression.decompress(compressedData);
  },

  getAvailableStrategies: function() {
    return JSON.stringify([
      { id: 'auto', name: 'Auto (Best)' },
      { id: 'pattern', name: 'Pattern Recognition' },
      { id: 'sequential', name: 'Sequential' },
      { id: 'spectral', name: 'Spectral' },
      { id: 'dictionary', name: 'Dictionary' }
    ]);
  }
};
```

### 2. Create a wrapper file for Emscripten

Create `primecompress-wasm-wrapper.js`:

```javascript
// This file serves as a wrapper for the Emscripten-compiled module
// It provides the glue code between the WebAssembly module and JavaScript

// The Module object will be created by Emscripten
var Module = {
  // Callback for when the module is ready
  onRuntimeInitialized: function() {
    console.log('PrimeCompress WebAssembly module initialized');
  },
  
  // Callback for when an error occurs
  onAbort: function(reason) {
    console.error('PrimeCompress WebAssembly module aborted:', reason);
  }
};

// Export the Module object
module.exports = Module;
```

### 3. Compile with Emscripten

Run the following command to compile the PrimeCompress library to WebAssembly:

```bash
emcc -O3 \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS="['_malloc', '_free']" \
  -s EXPORTED_RUNTIME_METHODS="['cwrap', 'setValue', 'getValue']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s USE_ES6_IMPORT_META=1 \
  -s ENVIRONMENT='web,worker' \
  --pre-js primecompress-wasm-wrapper.js \
  -o primecompress.js \
  primecompress-wasm.js ../src/core/unified-compression.js ../src/core/compression-wrapper.js \
  ../src/core/prime-compression.js \
  ../src/strategies/improved-dictionary-compression.js \
  ../src/strategies/improved-pattern-compression.js \
  ../src/strategies/improved-sequential-compression.js \
  ../src/strategies/improved-spectral-compression.js \
  ../src/utils/improved-corruption-detection.js \
  ../src/utils/sequence-solver.js
```

### 4. Copy the output files

Copy the generated files to the frontend directory:

```bash
cp primecompress.js primecompress.wasm ../frontend/public/
```

## Using the WebAssembly Module

To use the compiled WebAssembly module, update `prime-compress-wasm.ts` to load and use the compiled module:

```typescript
// Import the Emscripten-generated module
import PrimeCompressWasmModule from '../../public/primecompress.js';

// ... rest of the implementation
```

The actual implementation will replace the mock functions with calls to the WebAssembly module.

## Debugging the WebAssembly Module

For debugging purposes, you can add the following flags to the Emscripten compilation:

```bash
-g4 -s ASSERTIONS=2 -s SAFE_HEAP=1
```

This will enable:
- Debug information in the WebAssembly module
- Runtime assertions
- Memory access checking

## Optimizing the WebAssembly Module

For production builds, consider the following optimizations:

1. Use `-O3` optimization level
2. Add `-s ELIMINATE_DUPLICATE_FUNCTIONS=1`
3. Add `-s AGGRESSIVE_VARIABLE_ELIMINATION=1`
4. Consider using `-s STANDALONE_WASM=1` for browsers that support it

## Advanced Features

For advanced features like streaming compilation, add:

```javascript
// In the wrapper code
var Module = {
  instantiateWasm: function(imports, successCallback) {
    WebAssembly.instantiateStreaming(fetch('primecompress.wasm'), imports)
      .then(function(output) {
        successCallback(output.instance);
      });
    return {};
  }
};
```