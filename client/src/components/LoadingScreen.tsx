export default function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-grafana-darker">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-grafana-primary"></div>
    </div>
  );
} 