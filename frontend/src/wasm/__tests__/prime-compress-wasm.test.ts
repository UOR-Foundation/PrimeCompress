import PrimeCompressWasm, { WasmStatus } from '../prime-compress-wasm';

// Simplified tests for the PrimeCompressWasm module
describe('PrimeCompressWasm', () => {
  beforeEach(() => {
    // Reset module state before each test
    jest.resetModules();
    jest.clearAllMocks();
    
    // Create a mock wasmModule for tests
    const mockWasmModule = {
      compress: jest.fn().mockReturnValue({
        compressedData: new Uint8Array([1, 2, 3]),
        compressionRatio: 2.5,
        strategy: 'auto',
        originalSize: 10,
        compressedSize: 4,
        compressionTime: 100
      }),
      decompress: jest.fn().mockReturnValue(new Uint8Array([1, 2, 3, 4, 5])),
      getAvailableStrategies: jest.fn().mockReturnValue([
        { id: 'auto', name: 'Auto (Best)' },
        { id: 'pattern', name: 'Pattern Recognition' }
      ])
    };
    
    // Force module to LOADED state and attach mock module
    // @ts-ignore - Accessing private property for test
    PrimeCompressWasm.status = WasmStatus.LOADED;
    // @ts-ignore - Set the mock module
    PrimeCompressWasm.wasmModule = mockWasmModule;
  });

  it('should have LOADED status after setup', () => {
    // Our setup forced the status to LOADED
    expect(PrimeCompressWasm.getStatus()).toBe(WasmStatus.LOADED);
  });

  it('should provide available compression strategies', async () => {
    // Load the module - force it to LOADED state for this test
    // @ts-ignore - Accessing private property for test
    PrimeCompressWasm.status = WasmStatus.LOADED;
    
    // Get available strategies
    const strategies = await PrimeCompressWasm.getAvailableStrategies();
    
    // Strategies should be an array of objects with id and name
    expect(Array.isArray(strategies)).toBe(true);
    expect(strategies.length).toBeGreaterThan(0);
    
    // Each strategy should have id and name properties
    strategies.forEach(strategy => {
      expect(strategy).toHaveProperty('id');
      expect(strategy).toHaveProperty('name');
    });
  });

  it('should have compress method with correct mock interface', async () => {
    // Force module to LOADED state for this test
    // @ts-ignore - Accessing private property for test
    PrimeCompressWasm.status = WasmStatus.LOADED;
    
    // Create test data
    const testData = new Uint8Array([1, 2, 3, 4, 5]);
    
    // Compress the data
    const result = await PrimeCompressWasm.compress(testData);
    
    // Verify the result has the expected properties
    expect(result).toHaveProperty('compressedData');
    expect(result).toHaveProperty('compressionRatio');
    expect(result).toHaveProperty('strategy');
    expect(result).toHaveProperty('originalSize');
    expect(result).toHaveProperty('compressedSize');
    expect(result).toHaveProperty('compressionTime');
  });
});