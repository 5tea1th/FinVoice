'use client';

import Lottie from 'lottie-react';
import { useEffect, useState } from 'react';

const LOTTIE_URL = 'https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6930dae98ed924a2ab62c6a3_Government.json';

export default function TrybankSvg() {
  const [animationData, setAnimationData] = useState<unknown>(null);

  useEffect(() => {
    fetch(LOTTIE_URL)
      .then(res => res.json())
      .then(setAnimationData)
      .catch(() => {});
  }, []);

  if (!animationData) {
    return (
      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ width: 40, height: 40, border: '3px solid var(--orange-dim)', borderTopColor: 'var(--orange)', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
      </div>
    );
  }

  return (
    <Lottie
      animationData={animationData}
      loop
      autoplay
      style={{ width: '100%', height: '100%', filter: 'hue-rotate(190deg) saturate(1.3) brightness(1.1)' }}
    />
  );
}
