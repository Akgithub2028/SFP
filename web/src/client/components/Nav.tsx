import { useLocation, Link } from "react-router-dom";

const LINKS = [
  { href: "/", label: "Leaderboard" },
  { href: "/submit", label: "Submit" },
  { href: "/about", label: "About" },
];

export default function Nav() {
  const { pathname } = useLocation();

  return (
    <header
      className="mb-8 border-b pb-4"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="flex items-baseline justify-between">
        <div className="flex items-baseline gap-6">
          <Link to="/" className="hover:no-underline" onClick={(e) => { if (pathname === "/") e.preventDefault(); }}>
            <h1
              className="text-2xl font-bold tracking-tight"
              style={{ color: "var(--text)" }}
            >
              sfp
            </h1>
          </Link>
          <nav className="flex gap-4">
            {LINKS.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                onClick={(e) => { if (pathname === link.href) e.preventDefault(); }}
                className="text-sm transition-colors"
                style={{
                  color:
                    pathname === link.href
                      ? "var(--text)"
                      : "var(--text-dim)",
                  fontWeight: pathname === link.href ? 600 : 400,
                }}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
        <a
          href="https://github.com/paradigmxyz/sfp"
          className="text-xs"
          style={{ color: "var(--text-dim)" }}
        >
          GitHub ↗
        </a>
      </div>
      <p className="mt-1 text-xs" style={{ color: "var(--text-dim)" }}>
        An open benchmark for catastrophic forgetting in LLMs
      </p>
    </header>
  );
}
