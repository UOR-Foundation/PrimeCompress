# PrimeCompress Frontend

This is the React frontend for the PrimeCompress project, which provides a browser-based interface for compression using the PrimeCompress WebAssembly module.

## Features

- Client-side file compression
- Multiple compression strategies
- WebAssembly-based compression for performance
- Web Worker implementation for non-blocking UI
- Responsive Material UI design

## Architecture

The frontend is built with:
- React with TypeScript
- Material UI for components
- React Router for navigation
- WebAssembly for the compression engine
- Web Workers for background processing

## Development

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm start
```

### Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### Building

```bash
# Build for production
npm run build
```

## GitHub Pages Deployment

The application is configured for automatic deployment to GitHub Pages:

```bash
# Manual deployment
npm run deploy
```

The application is also set up with a GitHub Action workflow that automatically deploys to GitHub Pages on pushes to the main branch.

## WebAssembly Integration

The PrimeCompress library integration is designed to use WebAssembly with a singleton module pattern:

1. Currently uses a mock implementation that will be replaced with actual WebAssembly bindings
2. Compression operations run in a Web Worker to prevent UI blocking
3. The application can handle large files efficiently
4. Multiple compression strategies are available:
   - Pattern Recognition
   - Sequential
   - Spectral
   - Dictionary
   - Auto (selects best strategy)

See the `/src/wasm` directory for implementation details. The transition from mock to real WebAssembly will be seamless as all interfaces are already defined.

### Implementing Real WebAssembly

To replace the mock implementation with real WebAssembly:

1. Compile the PrimeCompress C/C++ code to WebAssembly
2. Place the compiled `.wasm` file in the `/public` directory
3. Update `prime-compress-wasm.ts` to load the real WebAssembly module
4. Implement the actual compression/decompression logic using the WebAssembly exports

The service layer and UI components will continue to work without changes once the WebAssembly implementation is in place.

## Project Structure

- `/src/components`: React components
- `/src/hooks`: React custom hooks
- `/src/pages`: Page components
- `/src/services`: Service layer
- `/src/utils`: Utility functions
- `/src/wasm`: WebAssembly integration