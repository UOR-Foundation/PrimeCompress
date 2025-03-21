import { useState } from 'react';
import { compressFile, CompressionResult } from '../services/compression';
import { CompressionOptions } from '../wasm/prime-compress-wasm';

interface UseCompressionResult {
  compressFile: (file: File, options?: CompressionOptions) => Promise<CompressionResult>;
  isCompressing: boolean;
  error: string | null;
  resetState: () => void;
}

/**
 * Custom hook for file compression operations
 */
export const useCompression = (): UseCompressionResult => {
  const [isCompressing, setIsCompressing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resetState = () => {
    setError(null);
  };

  const handleCompressFile = async (file: File, options?: CompressionOptions): Promise<CompressionResult> => {
    setIsCompressing(true);
    setError(null);
    
    try {
      // Use the actual implementation with WebAssembly
      return await compressFile(file, options);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Compression failed: ${errorMessage}`);
      throw err;
    } finally {
      setIsCompressing(false);
    }
  };

  return {
    compressFile: handleCompressFile,
    isCompressing,
    error,
    resetState
  };
};

export default useCompression;