/**
 * Web Worker for compression operations
 * 
 * This worker handles compression and decompression in a separate thread
 * to avoid blocking the main UI thread. It uses the PrimeCompress WASM module.
 */

// Note: This is a TypeScript file, but it will be compiled to JavaScript
// and used as a web worker. The TS types are for development only.

import PrimeCompressWasm, { CompressionOptions } from './prime-compress-wasm';

// Message types
interface CompressMessage {
  type: 'compress';
  id: string;
  data: Uint8Array;
  options?: CompressionOptions;
}

interface DecompressMessage {
  type: 'decompress';
  id: string;
  data: Uint8Array;
}

interface LoadWasmMessage {
  type: 'loadWasm';
  id: string;
}

type WorkerMessage = CompressMessage | DecompressMessage | LoadWasmMessage;

// Response types
interface SuccessResponse {
  success: true;
  id: string;
  result: any;
}

interface ErrorResponse {
  success: false;
  id: string;
  error: string;
}

// Create a worker context
declare const self: {
  onmessage: (event: MessageEvent<WorkerMessage>) => void;
  postMessage: (message: any) => void;
};

// Handle incoming messages
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  
  try {
    switch (message.type) {
      case 'loadWasm':
        await handleLoadWasm(message);
        break;
      case 'compress':
        await handleCompress(message);
        break;
      case 'decompress':
        await handleDecompress(message);
        break;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    const response: ErrorResponse = {
      success: false,
      id: message.id,
      error: errorMessage
    };
    self.postMessage(response);
  }
};

/**
 * Load the WebAssembly module
 */
async function handleLoadWasm(message: LoadWasmMessage) {
  try {
    await PrimeCompressWasm.load();
    const response: SuccessResponse = {
      success: true,
      id: message.id,
      result: { loaded: true }
    };
    self.postMessage(response);
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    const response: ErrorResponse = {
      success: false,
      id: message.id,
      error: errorMessage
    };
    self.postMessage(response);
  }
}

/**
 * Handle compression request
 */
async function handleCompress(message: CompressMessage) {
  try {
    const result = await PrimeCompressWasm.compress(message.data, message.options);
    const response: SuccessResponse = {
      success: true,
      id: message.id,
      result
    };
    self.postMessage(response);
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    const response: ErrorResponse = {
      success: false,
      id: message.id,
      error: errorMessage
    };
    self.postMessage(response);
  }
}

/**
 * Handle decompression request
 */
async function handleDecompress(message: DecompressMessage) {
  try {
    const result = await PrimeCompressWasm.decompress(message.data);
    const response: SuccessResponse = {
      success: true,
      id: message.id,
      result
    };
    self.postMessage(response);
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    const response: ErrorResponse = {
      success: false,
      id: message.id,
      error: errorMessage
    };
    self.postMessage(response);
  }
}