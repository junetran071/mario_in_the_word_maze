<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍄 Mario's Text Analysis Castle 🍄</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Fredoka+One:wght@400&display=swap');
        
        :root {
            --mario-red: #E60012;
            --mario-blue: #0066CC;
            --mario-yellow: #FFD700;
            --luigi-green: #00A652;
            --block-brown: #8B4513;
            --fire-orange: #FF8C00;
            --poison-purple: #8A2BE2;
            --sky-blue: #87CEEB;
            --cloud-white: #FFFFFF;
            --pipe-green: #228B22;
            --coin-gold: #FFD700;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Fredoka One', cursive;
            background: linear-gradient(135deg, var(--sky-blue) 0%, #98D8E8 50%, var(--sky-blue) 100%);
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* Animated Clouds */
        .cloud {
            position: absolute;
            background: var(--cloud-white);
            border-radius: 50px;
            opacity: 0.8;
            animation: float 20s infinite linear;
        }

        .cloud:before, .cloud:after {
            content: '';
            position: absolute;
            background: var(--cloud-white);
            border-radius: 50px;
        }

        .cloud1 {
            width: 100px;
            height: 40px;
            top: 20%;
            left: -100px;
            animation-duration: 25s;
        }

        .cloud1:before {
            width: 50px;
            height: 50px;
            top: -25px;
            left: 10px;
        }

        .cloud1:after {
            width: 60px;
            height: 40px;
            top: -15px;
            right: 10px;
        }

        .cloud2 {
            width: 80px;
            height: 30px;
            top: 40%;
            left: -80px;
            animation-duration: 30s;
            animation-delay: -10s;
        }

        .cloud2:before {
            width: 40px;
            height: 40px;
            top: -20px;
            left: 15px;
        }

        .cloud2:after {
            width: 50px;
            height: 30px;
            top: -10px;
            right: 15px;
        }

        .cloud3 {
            width: 120px;
            height: 50px;
            top: 60%;
            left: -120px;
            animation-duration: 35s;
            animation-delay: -5s;
        }

        .cloud3:before {
            width: 60px;
            height: 60px;
            top: -30px;
            left: 20px;
        }

        .cloud3:after {
            width: 70px;
            height: 50px;
            top: -20px;
            right: 20px;
        }

        @keyframes float {
            0% { left: -200px; }
            100% { left: 100vw; }
        }

        /* Mario Blocks */
        .block {
            position: absolute;
            width: 60px;
            height: 60px;
            background: var(--block-brown);
            border: 4px solid #654321;
            border-radius: 8px;
            animation: bounce 3s infinite ease-in-out;
        }

        .block1 { top: 15%; right: 10%; animation-delay: -0.5s; }
        .block2 { top: 35%; right: 5%; animation-delay: -1s; }
        .block3 { top: 55%; right: 15%; animation-delay: -1.5s; }

        @keyframes bounce {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
        }

        /* Main Container */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 10;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 50px;
            animation: slideDown 1s ease-out;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-50px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .main-title {
            font-family: 'Press Start 2P', cursive;
            font-size: clamp(24px, 5vw, 48px);
            color: var(--mario-red);
            text-shadow: 4px 4px 0px var(--cloud-white), 8px 8px 0px #000;
            margin-bottom: 20px;
            line-height: 1.2;
        }

        .subtitle {
            font-size: clamp(18px, 3vw, 32px);
            color: var(--mario-blue);
            text-shadow: 2px 2px 0px var(--cloud-white), 4px 4px 0px #000;
            margin-bottom: 30px;
        }

        /* Castle Structure */
        .castle {
            background: linear-gradient(145deg, #B8860B, #DAA520);
            border: 6px solid #8B4513;
            border-radius: 20px;
            padding: 40px;
            margin: 20px auto;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            position: relative;
            animation: castleAppear 1.5s ease-out 0.5s both;
        }

        @keyframes castleAppear {
            from { opacity: 0; transform: scale(0.8) translateY(50px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        /* Castle Towers */
        .castle:before, .castle:after {
            content: '';
            position: absolute;
            width: 80px;
            height: 120px;
            background: linear-gradient(145deg, #B8860B, #DAA520);
            border: 4px solid #8B4513;
            border-radius: 15px 15px 0 0;
            top: -60px;
        }

        .castle:before {
            left: 20px;
        }

        .castle:after {
            right: 20px;
        }

        /* Welcome Content */
        .welcome-content {
            text-align: center;
            color: var(--cloud-white);
            text-shadow: 2px 2px 0px #000;
        }

        .welcome-title {
            font-size: clamp(24px, 4vw, 40px);
            margin-bottom: 30px;
            color: var(--mario-red);
            text-shadow: 3px 3px 0px var(--cloud-white), 6px 6px 0px #000;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            margin: 40px 0;
        }

        .feature-card {
            background: var(--cloud-white);
            border: 4px solid var(--mario-blue);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            color: #333;
            transition: all 0.3s ease;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            animation: cardFloat 2s infinite ease-in-out;
        }

        .feature-card:nth-child(1) { animation-delay: -0.5s; }
        .feature-card:nth-child(2) { animation-delay: -1s; }
        .feature-card:nth-child(3) { animation-delay: -1.5s; }

        @keyframes cardFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .feature-card:hover {
            transform: translateY(-15px) scale(1.05);
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        }

        .feature-icon {
            font-size: 48px;
            margin-bottom: 15px;
            display: block;
        }

        .feature-title {
            font-size: 20px;
            font-weight: bold;
            color: var(--mario-red);
            margin-bottom: 10px;
        }

        .feature-description {
            font-size: 16px;
            line-height: 1.4;
            color: #555;
        }

        /* Action Buttons */
        .action-section {
            text-align: center;
            margin-top: 50px;
        }

        .mario-button {
            display: inline-block;
            background: linear-gradient(145deg, var(--mario-red), #CC0010);
            color: var(--cloud-white);
            padding: 18px 40px;
            border: 4px solid #8B0000;
            border-radius: 50px;
            font-family: 'Fredoka One', cursive;
            font-size: 20px;
            font-weight: bold;
            text-decoration: none;
            text-shadow: 2px 2px 0px #000;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            margin: 10px;
            position: relative;
            overflow: hidden;
        }

        .mario-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
            background: linear-gradient(145deg, #FF0015, var(--mario-red));
        }

        .mario-button:active {
            transform: translateY(-2px);
        }

        .luigi-button {
            background: linear-gradient(145deg, var(--luigi-green), #008B42);
            border-color: #006400;
        }

        .luigi-button:hover {
            background: linear-gradient(145deg, #00C652, var(--luigi-green));
        }

        /* Coins Animation */
        .coins {
            position: absolute;
            top: 10%;
            left: 5%;
            animation: coinSpin 4s infinite linear;
        }

        .coin {
            width: 40px;
            height: 40px;
            background: var(--coin-gold);
            border: 3px solid #B8860B;
            border-radius: 50%;
            display: inline-block;
            margin: 5px;
            position: relative;
            animation: coinBounce 2s infinite ease-in-out;
        }

        .coin:before {
            content: '★';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #B8860B;
            font-weight: bold;
            font-size: 20px;
        }

        @keyframes coinSpin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes coinBounce {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }

        /* Pipes */
        .pipe {
            position: absolute;
            bottom: 0;
            width: 80px;
            height: 150px;
            background: linear-gradient(90deg, var(--pipe-green), #32CD32);
            border: 4px solid #006400;
            border-radius: 10px 10px 0 0;
        }

        .pipe-left {
            left: 2%;
            animation: pipeGrow 2s ease-out 2s both;
        }

        .pipe-right {
            right: 2%;
            animation: pipeGrow 2s ease-out 2.5s both;
        }

        @keyframes pipeGrow {
            from { height: 0; }
            to { height: 150px; }
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .castle {
                padding: 20px;
                margin: 10px;
            }
            
            .feature-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .mario-button {
                padding: 15px 30px;
                font-size: 18px;
                display: block;
                margin: 15px auto;
                max-width: 280px;
            }
        }

        /* Power-up Effects */
        .power-up {
            position: absolute;
            font-size: 30px;
            animation: powerUpFloat 6s infinite ease-in-out;
        }

        .power-up1 {
            top: 25%;
            left: 8%;
            animation-delay: -1s;
        }

        .power-up2 {
            top: 70%;
            right: 8%;
            animation-delay: -3s;
        }

        @keyframes powerUpFloat {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-30px) rotate(120deg); }
            66% { transform: translateY(-15px) rotate(240deg); }
        }
    </style>
</head>
<body>
    <!-- Animated Background Elements -->
    <div class="cloud cloud1"></div>
    <div class="cloud cloud2"></div>
    <div class="cloud cloud3"></div>
    
    <div class="block block1"></div>
    <div class="block block2"></div>
    <div class="block block3"></div>
    
    <div class="coins">
        <div class="coin"></div>
        <div class="coin"></div>
        <div class="coin"></div>
    </div>
    
    <div class="power-up power-up1">🍄</div>
    <div class="power-up power-up2">⭐</div>
    
    <div class="pipe pipe-left"></div>
    <div class="pipe pipe-right"></div>

    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1 class="main-title">🍄 MARIO'S TEXT ANALYSIS CASTLE 🍄</h1>
            <p class="subtitle">Welcome to the Ultimate Text Analytics Adventure!</p>
        </header>

        <!-- Main Castle Content -->
        <div class="castle">
            <div class="welcome-content">
                <h2 class="welcome-title">🎯 Power-Up Your Text Analysis! 🎯</h2>
                
                <p style="font-size: 18px; margin-bottom: 30px; line-height: 1.6;">
                    Step into Mario's magical castle where text analysis meets gaming fun! 
                    Our powerful tools will help you discover hidden patterns, analyze marketing tactics, 
                    and unlock insights from your data with the power of the Mushroom Kingdom!
                </p>

                <!-- Features Grid -->
                <div class="feature-grid">
                    <div class="feature-card">
                        <span class="feature-icon">📊</span>
                        <h3 class="feature-title">Dictionary Analysis</h3>
                        <p class="feature-description">
                            Analyze text using custom dictionaries for urgency marketing, 
                            exclusive offers, and more tactical language detection.
                        </p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">🎨</span>
                        <h3 class="feature-title">Mario-Themed Visuals</h3>
                        <p class="feature-description">
                            Beautiful, colorful charts and graphs using Mario's iconic 
                            color palette to make data analysis fun and engaging.
                        </p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">⚡</span>
                        <h3 class="feature-title">Colab-Ready Tools</h3>
                        <p class="feature-description">
                            Minimalist Python code optimized for Google Colab with 
                            easy file handling and instant visualization capabilities.
                        </p>
                    </div>
                </div>

                <!-- Action Section -->
                <div class="action-section">
                    <h3 style="font-size: 24px; margin-bottom: 30px; color: var(--mario-yellow);">
                        🚀 Ready to Start Your Adventure? 🚀
                    </h3>
                    
                    <a href="#" class="mario-button" onclick="startAnalysis()">
                        🍄 Start Text Analysis
                    </a>
                    
                    <a href="#" class="mario-button luigi-button" onclick="viewDemo()">
                        👀 View Demo
                    </a>
                </div>

                <!-- Instructions -->
                <div style="margin-top: 40px; padding: 20px; background: rgba(255,255,255,0.9); border-radius: 15px; border: 3px solid var(--mario-blue);">
                    <h4 style="color: var(--mario-red); font-size: 20px; margin-bottom: 15px;">🎮 How to Play:</h4>
                    <ol style="text-align: left; color: #333; font-size: 16px; line-height: 1.6;">
                        <li><strong>Upload</strong> your CSV file to Google Colab</li>
                        <li><strong>Run</strong> the Mario Text Analysis script</li>
                        <li><strong>Watch</strong> as Mario analyzes your text data</li>
                        <li><strong>Enjoy</strong> beautiful, themed visualizations</li>
                        <li><strong>Download</strong> your results and level up!</li>
                    </ol>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer style="text-align: center; margin-top: 50px; padding: 20px;">
            <p style="color: var(--cloud-white); text-shadow: 2px 2px 0px #000; font-size: 18px;">
                🎉 Thank you Mario! Your text analysis adventure awaits! 🎉
            </p>
        </footer>
    </div>

    <script>
        function startAnalysis() {
            alert("🍄 Mario says: Upload your CSV file to Google Colab and run the analysis script! Let's-a-go!");
        }

        function viewDemo() {
            alert("👀 Demo coming soon! Mario is still working on the demo castle! 🏰");
        }

        // Add some interactive sparkles
        document.addEventListener('mousemove', function(e) {
            if (Math.random() > 0.9) {
                createSparkle(e.clientX, e.clientY);
            }
        });

        function createSparkle(x, y) {
            const sparkle = document.createElement('div');
            sparkle.innerHTML = '✨';
            sparkle.style.position = 'fixed';
            sparkle.style.left = x + 'px';
            sparkle.style.top = y + 'px';
            sparkle.style.fontSize = '20px';
            sparkle.style.pointerEvents = 'none';
            sparkle.style.zIndex = '1000';
            sparkle.style.animation = 'sparkleAnimation 1s ease-out forwards';
            
            const style = document.createElement('style');
            style.textContent = `
                @keyframes sparkleAnimation {
                    0% { opacity: 1; transform: scale(0) rotate(0deg); }
                    50% { opacity: 1; transform: scale(1) rotate(180deg); }
                    100% { opacity: 0; transform: scale(0) rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
            
            document.body.appendChild(sparkle);
            
            setTimeout(() => {
                document.body.removeChild(sparkle);
                document.head.removeChild(style);
            }, 1000);
        }

        // Coin collection sound effect simulation
        document.querySelectorAll('.coin').forEach(coin => {
            coin.addEventListener('click', function() {
                this.style.animation = 'none';
                this.style.transform = 'scale(1.5)';
                this.style.opacity = '0';
                setTimeout(() => {
                    this.style.animation = 'coinBounce 2s infinite ease-in-out';
                    this.style.transform = 'scale(1)';
                    this.style.opacity = '1';
                }, 500);
            });
        });
    </script>
</body>
</html>
