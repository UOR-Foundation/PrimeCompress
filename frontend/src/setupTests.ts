// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Helper for creating React elements in mocks
const mockReactElement = (type: any, props: any = null, ...children: any[]) => ({
  type,
  props: { ...props, children: children.length === 0 ? null : children.length === 1 ? children[0] : children },
  key: null,
  ref: null,
});

// Mock the MUI components as needed
jest.mock('@mui/material', () => {
  const actual = jest.requireActual('@mui/material');
  return {
    ...actual,
    Dialog: function Dialog({ children, open, onClose }: any) {
      return open ? mockReactElement('div', { 'data-testid': 'mock-dialog' }, children) : null;
    },
    DialogTitle: function DialogTitle({ children }: any) {
      return mockReactElement('h2', null, children);
    },
    DialogContent: function DialogContent({ children }: any) {
      return mockReactElement('div', null, children);
    },
    DialogActions: function DialogActions({ children }: any) {
      return mockReactElement('div', null, children);
    },
    Select: function Select({ children, onChange, value, label, labelId, disabled }: any) {
      return mockReactElement(
        'div',
        { 'data-testid': labelId || 'mock-select', 'aria-label': label },
        mockReactElement('select', { value, onChange, disabled }, children)
      );
    },
    MenuItem: function MenuItem({ children, value }: any) {
      return mockReactElement('option', { value }, children);
    },
    InputLabel: function InputLabel({ children, id }: any) {
      return mockReactElement('label', { htmlFor: id }, children);
    },
    FormControl: function FormControl({ children, ...props }: any) {
      return mockReactElement('div', props, children);
    },
    Button: function Button({ children, onClick, disabled, ...props }: any) {
      return mockReactElement('button', { onClick, disabled, ...props }, children);
    },
    CircularProgress: function CircularProgress(props: any) {
      return mockReactElement('div', { 'data-testid': 'mock-circular-progress' });
    },
    LinearProgress: function LinearProgress(props: any) {
      return mockReactElement('div', { 'data-testid': 'mock-linear-progress' });
    },
    Box: function Box({ children, ...props }: any) {
      return mockReactElement('div', props, children);
    },
    Typography: function Typography({ children, variant, ...props }: any) {
      const Element = variant === 'h6' ? 'h6' : 'p';
      return mockReactElement(Element, props, children);
    },
    Alert: function Alert({ children, severity, ...props }: any) {
      return mockReactElement('div', { role: 'alert', 'data-severity': severity, ...props }, children);
    },
    Paper: function Paper({ children, ...props }: any) {
      return mockReactElement('div', props, children);
    },
  };
});

// Mock WebAssembly
global.WebAssembly = {
  instantiateStreaming: jest.fn(),
  instantiate: jest.fn(),
  compile: jest.fn(),
  Module: jest.fn(),
  Instance: jest.fn(),
  Memory: jest.fn(),
} as any;

// Mock Web Workers
(global as any).Worker = class MockWorker {
  url: string | URL;
  options?: WorkerOptions;
  onmessage: ((event: MessageEvent) => void) | null = null;
  
  constructor(url: string | URL, options?: WorkerOptions) {
    this.url = url;
    this.options = options;
  }
  
  postMessage(message: any) {
    // Simulate successful response after a short delay
    setTimeout(() => {
      if (this.onmessage) {
        const responseEvent = {
          data: {
            success: true,
            id: message.id,
            result: { mockedResult: true }
          }
        } as MessageEvent;
        this.onmessage(responseEvent);
      }
    }, 10);
  }
  
  terminate() {
    // No-op for tests
  }
};

// Mock URL.createObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-blob-url');
global.URL.revokeObjectURL = jest.fn();