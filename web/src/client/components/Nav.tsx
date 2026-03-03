import { useLocation, Link } from "react-router-dom";

export default function Nav() {
  const { pathname } = useLocation();

  return (
    <header
      className="mb-8 border-b pb-4"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="flex items-baseline justify-between">
        <div className="flex items-baseline gap-6">
          <Link to="/" className="hover:no-underline">
            <h1
              className="text-2xl font-bold tracking-tight"
              style={{ color: "var(--text)" }}
            >
              sfp
            </h1>
          </Link>
          <nav className="flex gap-4">
            <Link
              to="/submit"
              className="text-sm transition-colors"
              style={{
                color: pathname === "/submit" ? "var(--text)" : "var(--text-dim)",
                fontWeight: pathname === "/submit" ? 600 : 400,
              }}
            >
              Submit
            </Link>
            <Link
              to="/about"
              className="text-sm transition-colors"
              style={{
                color: pathname === "/about" ? "var(--text)" : "var(--text-dim)",
                fontWeight: pathname === "/about" ? 600 : 400,
              }}
            >
              Motivation
            </Link>
            <a
              href="https://github.com/paradigmxyz/sfp"
              className="text-sm"
              style={{ color: "var(--text-dim)" }}
            >
              GitHub ↗
            </a>
          </nav>
        </div>
      </div>
      <p className="mt-1 text-xs" style={{ color: "var(--text-dim)" }}>
        An open benchmark for catastrophic forgetting in LLMs
      </p>
    </header>
  );
}
