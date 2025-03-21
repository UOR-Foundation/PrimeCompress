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
        // In production environments
        // @ts-ignore
        const workerPath = '../wasm/compression.worker.ts';
        this.worker = new Worker(workerPath, { type: 'module' });
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
    if (!this.worker) {
      await this.initialize();
    }
    
    return this.sendMessage('compress', data, options);
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