'use client';

import { useEffect } from 'react';

export default function ScrollAnimator() {
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const parent = entry.target.parentElement;
          const siblings = parent ? Array.from(parent.querySelectorAll('.fade-up')) : [];
          const index = siblings.indexOf(entry.target as Element);
          const delay = index >= 0 ? index * 100 : 0;

          setTimeout(() => {
            entry.target.classList.add('visible');
          }, delay);

          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -60px 0px'
    });

    document.querySelectorAll('.fade-up').forEach(el => observer.observe(el));

    // Stagger animation observer
    const staggerObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          staggerObserver.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    document.querySelectorAll('.stagger').forEach(el => staggerObserver.observe(el));

    return () => {
      observer.disconnect();
      staggerObserver.disconnect();
    };
  }, []);

  return null;
}
