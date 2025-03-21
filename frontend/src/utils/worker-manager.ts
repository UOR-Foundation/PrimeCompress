/**
 * Worker Manager
 * 
 * This utility manages Web Worker communication with a promise-based interface.
 */

import { CompressionOptions } from '../wasm/prime-compress-wasm';

// Worker responses
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

type WorkerResponse = SuccessResponse | ErrorResponse;

// Map of message IDs to their resolve/reject functions
interface PendingRequest {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

/**
 * Worker Manager
 */
class WorkerManager {
  private worker: Worker | null = null;
  private pendingRequests: Map<string, PendingRequest> = new Map();
  private messageCounter: number = 0;
  
  /**
   * Check if the worker is initialized
   */
  public isInitialized(): boolean {
    return this.worker !== null;
  }
  
  /**
   * Initialize the worker
   */
  public initialize() {
    if (this.worker) return;
    
    try {
      // Create the worker - in tests this will use the mocked Worker
      // In production, we would use a different approach to create the Worker
      // that is compatible with the bundler
      // In test environment, use a simpler approach to avoid import.meta issues
      if (process.env.NODE_ENV === 'test') {
        // In test environment, the Worker class is already mocked
        this.worker = new Worker('mock-path');
      } else {
        // Always use the inline worker to avoid MIME type issues
        console.log('Creating inline Web Worker for compression operations');
        
        // Create an inline worker implementation that imports PrimeCompressWasm
        const workerBlob = new Blob([`
          // Get the current script base URL to correctly import PrimeCompressWasm
          const scriptPath = self.location.href.substring(0, self.location.href.lastIndexOf('/'));
          const wasmPath = scriptPath + '/../wasm/prime-compress-wasm.js';
          
          // Import PrimeCompressWasm as a module
          importScripts(wasmPath);

          // The WebAssembly module will be available as PrimeCompressWasm
          let primeCompressWasm = null;

          self.onmessage = async function(event) {
            const message = event.data;
            const id = message.id;
            
            // Determine which operation to perform
            const operation = message.type || 'unknown';
            try {
              let result;

              // Handle different operation types
              if (operation === 'loadWasm') {
                // Initialize the WebAssembly module
                if (!primeCompressWasm) {
                  // Wait for the module to be available
                  if (typeof PrimeCompressWasm !== 'undefined') {
                    primeCompressWasm = PrimeCompressWasm;
                    await primeCompressWasm.load();
                  } else {
                    throw new Error('PrimeCompressWasm module not found');
                  }
                }
                result = { loaded: true };
              } 
              else if (operation === 'compress') {
                // Ensure the module is loaded
                if (!primeCompressWasm) {
                  if (typeof PrimeCompressWasm !== 'undefined') {
                    primeCompressWasm = PrimeCompressWasm;
                    await primeCompressWasm.load();
                  } else {
                    throw new Error('PrimeCompressWasm module not found');
                  }
                }
                
                // Use real compression
                const data = message.data || new Uint8Array(0);
                result = await primeCompressWasm.compress(data, message.options);
              }
              else if (operation === 'decompress') {
                // Ensure the module is loaded
                if (!primeCompressWasm) {
                  if (typeof PrimeCompressWasm !== 'undefined') {
                    primeCompressWasm = PrimeCompressWasm;
                    await primeCompressWasm.load();
                  } else {
                    throw new Error('PrimeCompressWasm module not found');
                  }
                }
                
                // Use real decompression
                const compressedData = message.data || new Uint8Array(0);
                result = await primeCompressWasm.decompress(compressedData);
              }
              else if (operation === 'getAvailableStrategies') {
                // Ensure the module is loaded
                if (!primeCompressWasm) {
                  if (typeof PrimeCompressWasm !== 'undefined') {
                    primeCompressWasm = PrimeCompressWasm;
                    await primeCompressWasm.load();
                  } else {
                    throw new Error('PrimeCompressWasm module not found');
                  }
                }
                
                // Get real strategies
                result = await primeCompressWasm.getAvailableStrategies();
              }
              else {
                throw new Error('Unknown operation: ' + operation);
              }
              
              // Send success response
              self.postMessage({
                success: true,
                id: id,
                result: result
              });
            } catch (error) {
              // Send error response
              self.postMessage({
                success: false,
                id: id,
                error: 'Worker error: ' + (error.message || 'Unknown error')
              });
            }
          };
        `], { type: 'application/javascript' });
        
        // Create the worker from the blob
        const workerUrl = URL.createObjectURL(workerBlob);
        this.worker = new Worker(workerUrl);
        
        // Clean up the URL when the worker is terminated
        this.worker.addEventListener('error', (event) => {
          console.error('Worker error:', event);
        });
      }
      
      // Set up message handler
      this.worker.onmessage = this.handleWorkerMessage.bind(this);
      
      // Load the WASM module
      return this.sendMessage('loadWasm', null);
    } catch (err) {
      console.error('Failed to initialize worker:', err);
      throw err;
    }
  }
  
  /**
   * Compress data using the worker
   */
  public async compress(data: Uint8Array, options?: CompressionOptions) {
    try {
      if (!this.worker) {
        await this.initialize();
      }
      
      console.log(`Sending compression request to worker with ${data.length} bytes`);
      return this.sendMessage('compress', data, options);
    } catch (error) {
      console.error('Worker compression failed, attempting direct compression:', error);
      
      // Try to use PrimeCompressWasm directly as a fallback
      try {
        // Dynamic import of PrimeCompressWasm
        const PrimeCompressWasm = await import('../wasm/prime-compress-wasm').then(module => module.default);
        await PrimeCompressWasm.load();
        
        // Use the module directly
        return await PrimeCompressWasm.compress(data, options);
      } catch (directError) {
        console.error('Direct compression also failed:', directError);
        // In this case, throw the error instead of returning mock data
        const errorMessage = directError instanceof Error 
          ? directError.message 
          : String(directError);
        throw new Error(`Compression failed: ${errorMessage || 'Unknown error'}`);
      }
    }
  }
  
  /**
   * Decompress data using the worker
   */
  public async decompress(data: Uint8Array) {
    if (!this.worker) {
      await this.initialize();
    }
    
    return this.sendMessage('decompress', data);
  }
  
  /**
   * Get available compression strategies
   */
  public async getAvailableStrategies() {
    if (!this.worker) {
      await this.initialize();
    }
    
    return this.sendMessage('getAvailableStrategies', null);
  }
  
  /**
   * Send a message to the worker
   */
  private sendMessage(type: string, data: any, options?: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error('Worker not initialized'));
        return;
      }
      
      // Generate a unique ID for this message
      const id = `${type}-${this.messageCounter++}`;
      
      // Store the resolve/reject functions
      this.pendingRequests.set(id, { resolve, reject });
      
      // Send the message to the worker
      this.worker.postMessage({
        type,
        id,
        data,
        options
      });
    });
  }
  
  /**
   * Handle responses from the worker
   */
  private handleWorkerMessage(event: MessageEvent<WorkerResponse>) {
    const response = event.data;
    const id = response.id;
    
    // Look up the pending request
    const pendingRequest = this.pendingRequests.get(id);
    if (!pendingRequest) {
      console.warn(`Received response for unknown request: ${id}`);
      return;
    }
    
    // Remove it from the pending map
    this.pendingRequests.delete(id);
    
    // Resolve or reject the promise
    if (response.success) {
      pendingRequest.resolve(response.result);
    } else {
      pendingRequest.reject(new Error(response.error));
    }
  }
  
  /**
   * Clean up the worker
   */
  public terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      
      // Reject all pending requests
      const requests = Array.from(this.pendingRequests.values());
      for (const request of requests) {
        request.reject(new Error('Worker terminated'));
      }
      
      this.pendingRequests.clear();
    }
  }
}

// Export a singleton instance
const workerManager = new WorkerManager();
export default workerManager;