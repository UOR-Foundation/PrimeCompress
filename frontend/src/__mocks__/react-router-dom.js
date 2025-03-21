// Mock for react-router-dom
const reactRouterDom = jest.createMockFromModule('react-router-dom');

// Mock Link component
reactRouterDom.Link = ({ to, children, ...props }) => {
  return (
    <a href={to} data-testid="nav-link" {...props}>
      {children}
    </a>
  );
};

// Mock useLocation
reactRouterDom.useLocation = jest.fn().mockReturnValue({ pathname: '/' });

// Mock useNavigate
reactRouterDom.useNavigate = jest.fn().mockReturnValue(jest.fn());

// Mock useParams
reactRouterDom.useParams = jest.fn().mockReturnValue({});

// Mock Routes and Route
reactRouterDom.Routes = ({ children }) => <div data-testid="routes">{children}</div>;
reactRouterDom.Route = ({ path, element }) => <div data-testid={`route-${path}`}>{element}</div>;

// Mock BrowserRouter
reactRouterDom.BrowserRouter = ({ children }) => <div data-testid="browser-router">{children}</div>;

module.exports = reactRouterDom;