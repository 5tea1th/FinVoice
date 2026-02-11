'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav className={`nav${scrolled ? ' scrolled' : ''}`}>
      <div className="nav-inner container">
        <Link href="/" className="nav-logo">
          FIN<span>VOICE</span>
        </Link>
        <div className="nav-links">
          <a href="#pipeline" className="nav-link">Pipeline</a>
          <a href="#features" className="nav-link">Features</a>
          <a href="#infra" className="nav-link">Infrastructure</a>
        </div>
        <Link href="/dashboard" className="btn btn-primary">
          Launch Dashboard
        </Link>
      </div>
    </nav>
  );
}
