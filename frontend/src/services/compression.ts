/**
 * Compression Service
 * 
 * This service provides an interface to the PrimeCompress library
 * using WebAssembly bindings and Web Workers for non-blocking operations.
 */

import workerManager from '../utils/worker-manager';
import { CompressionOptions } from '../wasm/prime-compress-wasm';

// Interface for compression result
export interface CompressionResult {
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  strategy: string;
  compressionTime: number;
  compressedBlob?: Blob;
}

/**
 * Compress a file using PrimeCompress
 * 
 * @param file The file to compress
 * @param options Compression options
 * @returns Promise with compression result
 */
export const compressFile = async (
  file: File, 
  options: CompressionOptions = {}
): Promise<CompressionResult> => {
  try {
    // Read the file as an ArrayBuffer
    const arrayBuffer = await file.arrayBuffer();
    const data = new Uint8Array(arrayBuffer);
    
    // Start timer to measure performance
    const startTime = performance.now();
    
    // Compress the data using the worker for non-blocking operation
    const result = await workerManager.compress(data, options);
    
    // Calculate time taken
    const compressionTime = performance.now() - startTime;
    
    // Create a blob for the compressed data
    const compressedBlob = new Blob([result.compressedData], { type: 'application/octet-stream' });
    
    return {
      originalSize: result.originalSize,
      compressedSize: result.compressedSize,
      compressionRatio: result.compressionRatio,
      strategy: result.strategy,
      compressionTime,
      compressedBlob
    };
  } catch (err) {
    console.error('Compression error:', err);
    throw err;
  }
};

/**
 * Legacy function maintained for compatibility with existing code
 * Now just calls the real compressFile function directly
 */
export const mockCompressFile = async (file: File) => {
  return await compressFile(file);
};

/**
 * Test compression with different strategies
 * 
 * @param data Data to compress
 * @param strategy Compression strategy to use
 * @returns Promise with compression result
 */
export const testCompression = async (
  data: ArrayBuffer, 
  strategy?: string
): Promise<CompressionResult> => {
  try {
    console.log(`Running compression test with strategy: ${strategy || 'auto'}`);
    
    // Convert ArrayBuffer to Uint8Array for the compression
    const uint8Data = new Uint8Array(data);
    
    // Build compression options
    const options: CompressionOptions = {};
    if (strategy) {
      options.strategy = strategy;
    }
    
    // Use the worker for non-blocking compression
    const result = await workerManager.compress(uint8Data, options);
    
    return {
      originalSize: result.originalSize,
      compressedSize: result.compressedSize,
      compressionRatio: result.compressionRatio,
      strategy: result.strategy,
      compressionTime: result.compressionTime
    };
  } catch (err) {
    console.error('Test compression error:', err);
    throw err;
  }
};

/**
 * Define Strategy interface
 */
export interface Strategy {
  id: string;
  name: string;
}

/**
 * Get a list of available compression strategies
 */
export const getAvailableStrategies = async (): Promise<Strategy[]> => {
  try {
    // Get strategies through the worker manager
    console.log('Getting available compression strategies from the WASM module');
    return await workerManager.getAvailableStrategies();
  } catch (err) {
    console.error('Error getting available strategies:', err);
    throw err;
  }
};