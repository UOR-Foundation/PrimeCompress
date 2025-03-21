import { renderHook, act } from '@testing-library/react';
import useCompression from '../useCompression';
import * as compressionService from '../../services/compression';

// Mock the compression service
jest.mock('../../services/compression', () => ({
  compressFile: jest.fn(),
}));

describe('useCompression hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  it('should return initial state', () => {
    const { result } = renderHook(() => useCompression());
    
    expect(result.current.isCompressing).toBe(false);
    expect(result.current.error).toBeNull();
    expect(typeof result.current.compressFile).toBe('function');
    expect(typeof result.current.resetState).toBe('function');
  });
  
  it('should handle successful compression', async () => {
    // Mock successful compression
    const mockResult = {
      originalSize: 1000,
      compressedSize: 500,
      compressionRatio: 2,
      strategy: 'auto',
      compressionTime: 150,
      compressedBlob: new Blob(['test'], { type: 'application/octet-stream' })
    };
    (compressionService.compressFile as jest.Mock).mockResolvedValue(mockResult);
    
    const { result } = renderHook(() => useCompression());
    
    // Mock file
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    
    // Call the hook method
    let compressResult;
    await act(async () => {
      compressResult = await result.current.compressFile(file);
    });
    
    // Check service was called
    expect(compressionService.compressFile).toHaveBeenCalledWith(file, undefined);
    
    // Check state changes
    expect(result.current.isCompressing).toBe(false);
    expect(result.current.error).toBeNull();
    
    // Check result
    expect(compressResult).toEqual(mockResult);
  });
  
  it('should handle compression with options', async () => {
    // Mock successful compression
    const mockResult = {
      originalSize: 1000,
      compressedSize: 500,
      compressionRatio: 2,
      strategy: 'pattern',
      compressionTime: 150,
      compressedBlob: new Blob(['test'], { type: 'application/octet-stream' })
    };
    (compressionService.compressFile as jest.Mock).mockResolvedValue(mockResult);
    
    const { result } = renderHook(() => useCompression());
    
    // Mock file and options
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    const options = { strategy: 'pattern' };
    
    // Call the hook method
    let compressResult;
    await act(async () => {
      compressResult = await result.current.compressFile(file, options);
    });
    
    // Check service was called with options
    expect(compressionService.compressFile).toHaveBeenCalledWith(file, options);
    
    // Check result
    expect(compressResult).toEqual(mockResult);
  });
  
  it('should handle compression failure', async () => {
    // Mock compression failure
    const error = new Error('Compression failed');
    (compressionService.compressFile as jest.Mock).mockRejectedValue(error);
    
    const { result } = renderHook(() => useCompression());
    
    // Mock file
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    
    // Call the hook method
    await act(async () => {
      try {
        await result.current.compressFile(file);
      } catch (err) {
        // Expected error
      }
    });
    
    // Check state changes
    expect(result.current.isCompressing).toBe(false);
    expect(result.current.error).toBe('Compression failed: Compression failed');
  });
  
  it('should reset error state', async () => {
    // Mock compression failure to set an error
    const error = new Error('Compression failed');
    (compressionService.compressFile as jest.Mock).mockRejectedValue(error);
    
    const { result } = renderHook(() => useCompression());
    
    // Mock file
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    
    // Cause an error
    await act(async () => {
      try {
        await result.current.compressFile(file);
      } catch (err) {
        // Expected error
      }
    });
    
    // Verify error state
    expect(result.current.error).toBe('Compression failed: Compression failed');
    
    // Reset error state
    act(() => {
      result.current.resetState();
    });
    
    // Check error was reset
    expect(result.current.error).toBeNull();
  });
});