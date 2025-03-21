# PrimeCompress

A sophisticated compression library featuring multiple strategies for optimal compression across diverse data types. Built on mathematical principles and advanced compression techniques to achieve superior compression ratios for various data patterns.

## Try it Online

Visit our [PrimeCompress Web Application](https://uor-foundation.github.io/PrimeCompress/) to compress files directly in your browser using WebAssembly.

![PrimeCompress Web App](./frontend/public/logo192.png)

## Features

- **Strategy Selection**: Intelligent selection of compression strategy based on data characteristics
- **Block-Based Compression**: Splits larger data into blocks, applying optimal compression for each block
- **True Huffman Encoding**: Dictionary compression with Huffman coding for text data
- **Fast Path for High-Entropy Data**: Optimized handling of random or incompressible data
- **Pattern Detection**: Recognition of constants, patterns, sequences, and spectral characteristics
- **Perfect Reconstruction**: Guaranteed exact data reconstruction with checksum validation

## Compression Strategies

PrimeCompress uses multiple compression strategies:

- **Zeros**: For constant data (all bytes are the same)
- **Pattern**: For repeating patterns in binary data
- **Sequential**: For arithmetic sequences and i%N patterns
- **Spectral**: For sinusoidal and wave-like data
- **Dictionary**: For text-like data with frequency-optimized dictionary
- **Statistical**: Fallback for high-entropy data

## Performance

PrimeCompress delivers impressive compression ratios:

| Data Type | Improvement over Standard Compression |
|-----------|-------------------------------------|
| Text | +185.83% |
| Mixed Data (Block-Based) | +210.59% |
| Sine Wave | +60.00% |
| Sequential | Matches standard |
| Zeros | Matches standard |
| Random | Matches standard |

Overall average improvement: **40.97%**

### Hutter Prize Benchmark

Our compression performance is measured against the [Hutter Prize](http://prize.hutter1.net/) dataset (enwik8), a standard benchmark in the compression community:

[![Compression Ratio](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/UOR-Foundation/PrimeCompress/main/badges/hutter-ratio.json)](https://github.com/UOR-Foundation/PrimeCompress/actions/workflows/hutter-prize-benchmark.yml)
[![Bits Per Character](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/UOR-Foundation/PrimeCompress/main/badges/hutter-bpc.json)](https://github.com/UOR-Foundation/PrimeCompress/actions/workflows/hutter-prize-benchmark.yml)
[![Best Algorithm](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/UOR-Foundation/PrimeCompress/main/badges/hutter-algo.json)](https://github.com/UOR-Foundation/PrimeCompress/actions/workflows/hutter-prize-benchmark.yml)

You can run the benchmark yourself by triggering the "Hutter Prize Benchmark" workflow in the Actions tab.

## Installation

From GitHub Packages:

```bash
# Configure npm to use GitHub Packages
echo "@uor-foundation:registry=https://npm.pkg.github.com" >> .npmrc

# Install the package
npm install @uor-foundation/prime-compress
```

From source:

```bash
npm install
```

## Usage

```javascript
// When installed from npm
const compression = require('@uor-foundation/prime-compress');

// OR when using from source
// const compression = require('./unified-compression.js');

// Compress data
const compressedData = compression.compress(data);

// Decompress data
const originalData = compression.decompress(compressedData);

// Compress with a specific strategy
const compressedWithStrategy = compression.compressWithStrategy(data, 'dictionary');

// Analyze data to determine optimal compression strategy
const analysis = compression.analyzeCompression(data);
console.log(`Recommended strategy: ${analysis.recommendedStrategy}`);
```

## Advanced Features

### Block-Based Compression

For larger datasets, PrimeCompress automatically applies block-based compression, dividing data into optimal blocks and compressing each with the best strategy.

```javascript
// Enable block-based compression (enabled by default for data > 4KB)
const compressedBlocks = compression.compress(largeData, { useBlocks: true });
```

### Enhanced Dictionary Compression

Text data benefits from frequency-optimized dictionary compression with Huffman coding for further space savings.

```javascript
// Force dictionary strategy with Huffman coding
const compressedText = compression.compressWithStrategy(textData, 'enhanced-dictionary');
```

## Analysis Tools

PrimeCompress includes tools for analyzing data characteristics:

```javascript
const analysis = compression.analyzeCompression(data);
console.log(`Entropy: ${analysis.entropy}`);
console.log(`Recommended strategy: ${analysis.recommendedStrategy}`);
console.log(`Estimated compression ratio: ${analysis.theoreticalCompressionRatio}x`);
```

## Mathematical Basis

The implementation is based on advanced mathematical principles including:

- Entropy-based optimization for strategy selection
- Spectral analysis for wave-like data patterns
- Frequency distribution analysis for dictionary optimization
- Huffman tree construction for optimal prefix coding

## Testing

Run the comprehensive test suite:

```bash
node unified-compression-test.js
```

## Web Application

The PrimeCompress project includes a React-based web application that allows users to compress files directly in their browser using WebAssembly. The web application:

- Runs completely client-side (no server required)
- Uses WebAssembly for high-performance compression
- Processes files in a Web Worker to prevent UI blocking
- Provides multiple compression strategies
- Shows detailed compression statistics

The frontend code is located in the `./frontend` directory. See the [Frontend README](./frontend/README.md) for more details.

### Development

```bash
cd frontend
npm install
npm start
```

### Deployment

The web application is automatically deployed to GitHub Pages when changes are pushed to the main branch. You can also deploy it manually:

```bash
cd frontend
npm run deploy
```

## Publishing

This package is configured to publish to GitHub Packages. 

### Automatic Publishing (CI/CD)

The package will automatically be published to GitHub Packages when a new release is created on GitHub.

### Manual Publishing

To publish manually:

1. Log in to the GitHub Packages registry:
   ```bash
   npm login --registry=https://npm.pkg.github.com
   ```

2. Update the version in package.json:
   ```bash
   npm version patch  # or minor, or major
   ```

3. Publish the package:
   ```bash
   npm publish
   ```

## License

MIT