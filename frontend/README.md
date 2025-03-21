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

The PrimeCompress library is compiled to WebAssembly and integrated using a singleton module pattern:

1. The WebAssembly module is loaded when the application starts
2. Compression operations run in a Web Worker to prevent UI blocking
3. The application can handle large files efficiently
4. Multiple compression strategies are available

See the `/src/wasm` directory for implementation details.

## Project Structure

- `/src/components`: React components
- `/src/hooks`: React custom hooks
- `/src/pages`: Page components
- `/src/services`: Service layer
- `/src/utils`: Utility functions
- `/src/wasm`: WebAssembly integration