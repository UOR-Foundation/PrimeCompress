# PrimeOS Development Guide

## Build Commands
- Build: `npm run build`, `npm run build:dev`, `npm run build:prod`
- Lint: `npm run lint` (ESLint for src/**/*.js)
- Format: `npm run format` (Prettier for src/ and tests/)
- Test: `npm run test` (unit tests), `npm run test:integration`, `npm run test:browser`, `npm run test:all`
- Docs: `npm run docs`, `npm run docs:dev`

## Frontend Development
### Build and Run
- Frontend Install: `cd frontend && npm install` 
- Frontend Start: `cd frontend && npm start`
- Frontend Build: `cd frontend && npm run build`
- Frontend Test: `cd frontend && npm test`
- Frontend Lint: `cd frontend && npm run lint`
- Frontend Deploy to GitHub Pages: `cd frontend && npm run deploy`

### Technology Stack
- React with TypeScript
- Material UI for components
- WebAssembly for compression
- Web Workers for non-blocking operations
- GitHub Pages for hosting

## Code Style Guidelines
- **Formatting**: Use Prettier with ESLint config prettier
- **Naming**: camelCase for variables/functions, PascalCase for classes
- **Documentation**: JSDoc comments for all public methods and classes
- **Error Handling**: Use custom error classes (PrimeError hierarchy)
- **Types**: Thorough type checking with Utils.isXXX() methods
- **Imports/Exports**: Module pattern for encapsulation, CommonJS compatibility
- **Testing**: Jest for unit tests, Puppeteer for browser tests
- **Environment**: Node.js >= 18

## Best Practices
- Validate input parameters with type checking
- Use provided utility methods for common operations
- Follow mathematical coherence principles
- Check version compatibility where needed
- Handle errors with appropriate error classes