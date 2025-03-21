import workerManager from '../worker-manager';

// Mock the Worker class
jest.mock('../../wasm/compression.worker.ts', () => 'mock-worker-url', { virtual: true });

// Only run simplified tests for worker-manager
describe('WorkerManager basic operations', () => {
  it('should export a singleton instance', () => {
    expect(workerManager).toBeDefined();
    expect(typeof workerManager.initialize).toBe('function');
    expect(typeof workerManager.compress).toBe('function');
    expect(typeof workerManager.decompress).toBe('function');
    expect(typeof workerManager.terminate).toBe('function');
  });
});

