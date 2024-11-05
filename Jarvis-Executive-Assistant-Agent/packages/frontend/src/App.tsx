import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mui/material';
import Layout from '@/components/Layout';
import PrivateRoute from '@/components/PrivateRoute';
import { useAuth } from '@/hooks/useAuth';

// Lazy load pages
import { lazy, Suspense } from 'react';
const Chat = lazy(() => import('@/pages/Chat'));
const Login = lazy(() => import('@/pages/Login'));
const Settings = lazy(() => import('@/pages/Settings'));
const Profile = lazy(() => import('@/pages/Profile'));
const NotFound = lazy(() => import('@/pages/NotFound'));

function App() {
  const { isInitialized } = useAuth();

  if (!isInitialized) {
    return (
      <Box className="flex-center full-height">
        {/* You can replace this with a proper loading component */}
        <div>Loading...</div>
      </Box>
    );
  }

  return (
    <Suspense
      fallback={
        <Box className="flex-center full-height">
          <div>Loading...</div>
        </Box>
      }
    >
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />

        {/* Protected routes */}
        <Route element={<PrivateRoute />}>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/chat/:conversationId" element={<Chat />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/profile" element={<Profile />} />
          </Route>
        </Route>

        {/* 404 route */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
}

export default App;
