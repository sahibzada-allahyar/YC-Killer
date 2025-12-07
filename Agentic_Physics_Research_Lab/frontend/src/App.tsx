import { ChatInterface } from './components/ChatInterface'

function App() {
  return (
    <div className="min-h-screen bg-tokyonight-bg bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-tokyonight-dark to-tokyonight-bg">
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>
      <ChatInterface />
    </div>
  )
}

export default App
