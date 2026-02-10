'use client';

import { use } from 'react';
import { useRouter } from 'next/navigation';
import CallDetail from '@/components/dashboard/CallDetail';

export default function CallPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  return <CallDetail callId={id} onBack={() => router.push('/dashboard')} />;
}
