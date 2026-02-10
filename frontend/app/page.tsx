import Navbar from '@/components/landing/Navbar';
import Hero from '@/components/landing/Hero';
import Pipeline from '@/components/landing/Pipeline';
import { Features, Backboard, Infrastructure, CTAFooter } from '@/components/landing/Sections';
import ScrollAnimator from '@/components/landing/ScrollAnimator';

export default function LandingPage() {
  return (
    <div>
      <Navbar />
      <Hero />
      <Pipeline />
      <Features />
      <Backboard />
      <Infrastructure />
      <CTAFooter />
      <ScrollAnimator />
    </div>
  );
}
