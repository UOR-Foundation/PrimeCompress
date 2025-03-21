/**
 * Comprehensive tests for the compression service
 */
import { compressFile, testCompression, getAvailableStrategies } from '../compression';
import workerManager from '../../utils/worker-manager';

// Mock the worker manager
jest.mock('../../utils/worker-manager', () => ({
  compress: jest.fn(),
  getAvailableStrategies: jest.fn(),
}));

describe('Compression Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('compressFile', () => {
    it('should correctly compress a file with default options', async () => {
      // Mock successful compression result
      const mockCompressResult = {
        compressedData: new Uint8Array([0xF0, 1, 2, 3]),
        originalSize: 1000,
        compressedSize: 500,
        compressionRatio: 2.0,
        strategy: 'auto',
        compressionTime: 150
      };
      
      (workerManager.compress as jest.Mock).mockResolvedValue(mockCompressResult);
      
      // Create a test file with arrayBuffer method mock
      const file = new File(['test'.repeat(250)], 'test.txt', { type: 'text/plain' });
      // Mock the arrayBuffer method
      file.arrayBuffer = jest.fn().mockResolvedValue(new ArrayBuffer(1000));
      
      // Compress the file
      const result = await compressFile(file);
      
      // Verify worker manager was called correctly
      expect(workerManager.compress).toHaveBeenCalledWith(expect.any(Uint8Array), {});
      
      // Verify result structure
      expect(result).toHaveProperty('originalSize', 1000);
      expect(result).toHaveProperty('compressedSize', 500);
      expect(result).toHaveProperty('compressionRatio', 2.0);
      expect(result).toHaveProperty('strategy', 'auto');
      expect(result).toHaveProperty('compressionTime');
      expect(result).toHaveProperty('compressedBlob');
      
      // Blob should contain the compressed data
      expect(result.compressedBlob instanceof Blob).toBe(true);
      expect(result.compressedBlob!.size).toBeGreaterThan(0);
      expect(result.compressedBlob!.type).toBe('application/octet-stream');
    });
    
    it('should correctly compress a file with specific strategy', async () => {
      // Mock successful compression result for pattern strategy
      const mockCompressResult = {
        compressedData: new Uint8Array([0xC0, 1, 2, 3]),
        originalSize: 1000,
        compressedSize: 400,
        compressionRatio: 2.5,
        strategy: 'pattern',
        compressionTime: 120
      };
      
      (workerManager.compress as jest.Mock).mockResolvedValue(mockCompressResult);
      
      // Create a test file (repetitive pattern for pattern compression)
      const file = new File(['ABABABABAB'.repeat(100)], 'pattern.txt', { type: 'text/plain' });
      // Mock the arrayBuffer method
      file.arrayBuffer = jest.fn().mockResolvedValue(new ArrayBuffer(1000));
      
      // Compress with pattern strategy
      const result = await compressFile(file, { strategy: 'pattern' });
      
      // Verify worker manager was called with correct options
      expect(workerManager.compress).toHaveBeenCalledWith(
        expect.any(Uint8Array),
        { strategy: 'pattern' }
      );
      
      // Verify result has pattern strategy
      expect(result.strategy).toBe('pattern');
      expect(result.compressionRatio).toBe(2.5);
    });
    
    it('should handle very large files', async () => {
      // Mock compression result for large file
      const mockCompressResult = {
        compressedData: new Uint8Array(100000), // 100KB compressed
        originalSize: 1000000, // 1MB original
        compressedSize: 100000,
        compressionRatio: 10.0,
        strategy: 'dictionary',
        compressionTime: 500
      };
      
      (workerManager.compress as jest.Mock).mockResolvedValue(mockCompressResult);
      
      // Create a large test file (1MB)
      const largeArray = new Uint8Array(1000000);
      const file = new File([largeArray], 'large.bin', { type: 'application/octet-stream' });
      // Mock the arrayBuffer method
      file.arrayBuffer = jest.fn().mockResolvedValue(new ArrayBuffer(1000000));
      
      // Compress the file
      const result = await compressFile(file, { strategy: 'dictionary' });
      
      // Verify compression of large file
      expect(result.originalSize).toBe(1000000);
      expect(result.compressedSize).toBe(100000);
      expect(result.compressionRatio).toBe(10.0);
      expect(result.compressionTime).toBeGreaterThan(0);
    });
    
    it('should handle compression errors properly', async () => {
      // Mock compression failure
      const error = new Error('Compression error: out of memory');
      (workerManager.compress as jest.Mock).mockRejectedValue(error);
      
      // Create a test file
      const file = new File(['test'], 'error.txt', { type: 'text/plain' });
      // Mock the arrayBuffer method
      file.arrayBuffer = jest.fn().mockResolvedValue(new ArrayBuffer(4));
      
      // Attempt to compress
      await expect(compressFile(file)).rejects.toThrow('Compression error: out of memory');
    });
  });
  
  describe('testCompression', () => {
    it('should test compression with auto strategy', async () => {
      // Mock compression result
      const mockCompressResult = {
        compressedData: new Uint8Array([0xF0, 1, 2, 3]),
        originalSize: 100,
        compressedSize: 50,
        compressionRatio: 2.0,
        strategy: 'auto',
        compressionTime: 50
      };
      
      (workerManager.compress as jest.Mock).mockResolvedValue(mockCompressResult);
      
      // Create test data
      const testData = new ArrayBuffer(100);
      
      // Test compression
      const result = await testCompression(testData);
      
      // Verify worker manager was called correctly
      expect(workerManager.compress).toHaveBeenCalledWith(expect.any(Uint8Array), {});
      
      // Verify result
      expect(result.strategy).toBe('auto');
      expect(result.originalSize).toBe(100);
      expect(result.compressedSize).toBe(50);
    });
    
    it('should test compression with specific strategy', async () => {
      // Mock compression result
      const mockCompressResult = {
        compressedData: new Uint8Array([0xC3, 1, 2, 3]),
        originalSize: 100,
        compressedSize: 40,
        compressionRatio: 2.5,
        strategy: 'spectral',
        compressionTime: 60
      };
      
      (workerManager.compress as jest.Mock).mockResolvedValue(mockCompressResult);
      
      // Create test data
      const testData = new ArrayBuffer(100);
      
      // Test compression with spectral strategy
      const result = await testCompression(testData, 'spectral');
      
      // Verify worker manager was called with correct options
      expect(workerManager.compress).toHaveBeenCalledWith(
        expect.any(Uint8Array),
        { strategy: 'spectral' }
      );
      
      // Verify result
      expect(result.strategy).toBe('spectral');
      expect(result.compressionRatio).toBe(2.5);
    });
  });
  
  describe('getAvailableStrategies', () => {
    it('should return all required compression strategies', async () => {
      // Mock available strategies
      const mockStrategies = [
        { id: 'auto', name: 'Auto (Best)' },
        { id: 'pattern', name: 'Pattern Recognition' },
        { id: 'sequential', name: 'Sequential Compression' },
        { id: 'dictionary', name: 'Dictionary Compression' },
        { id: 'spectral', name: 'Spectral Analysis' }
      ];
      
      (workerManager.getAvailableStrategies as jest.Mock).mockResolvedValue(mockStrategies);
      
      // Get available strategies
      const strategies = await getAvailableStrategies();
      
      // Verify all required strategies are present
      expect(strategies.length).toBe(5);
      
      // Check for each required strategy
      const requiredStrategies = ['auto', 'pattern', 'sequential', 'dictionary', 'spectral'];
      requiredStrategies.forEach(strategyId => {
        const strategy = strategies.find(s => s.id === strategyId);
        expect(strategy).toBeDefined();
        expect(strategy).toHaveProperty('name');
      });
    });
    
    it('should handle strategy retrieval errors', async () => {
      // Mock strategy retrieval failure
      const error = new Error('Failed to load strategies');
      (workerManager.getAvailableStrategies as jest.Mock).mockRejectedValue(error);
      
      // Attempt to get strategies
      await expect(getAvailableStrategies()).rejects.toThrow('Failed to load strategies');
    });
  });
});