const canvas = document.getElementById("game");
const context = canvas.getContext("2d");

const scoreEl = document.getElementById("score");
const highScoreEl = document.getElementById("highScore");
const overlay = document.getElementById("overlay");
const overlayText = document.getElementById("overlayText");
const pauseBtn = document.getElementById("pauseBtn");
const restartBtn = document.getElementById("restartBtn");
const touchPauseBtn = document.getElementById("touchPause");
const touchRestartBtn = document.getElementById("touchRestart");

const CELL_SIZE = 24;
const GRID_COLS = Math.floor(canvas.width / CELL_SIZE);
const GRID_ROWS = Math.floor(canvas.height / CELL_SIZE);

let snakeCells = [];
let foodCell = null;
let currentDirection = { x: 1, y: 0 };
let nextDirection = { x: 1, y: 0 };
let isRunning = false;
let isPaused = false;
let isGameOver = false;
let score = 0;
let highScore = Number.parseInt(localStorage.getItem("snakeHighScore") || "0", 10);
let tickId = null;
let tickIntervalMs = 120;

function initializeGame() {
  snakeCells = [
    { x: Math.floor(GRID_COLS / 2) + 1, y: Math.floor(GRID_ROWS / 2) },
    { x: Math.floor(GRID_COLS / 2), y: Math.floor(GRID_ROWS / 2) },
    { x: Math.floor(GRID_COLS / 2) - 1, y: Math.floor(GRID_ROWS / 2) }
  ];
  currentDirection = { x: 1, y: 0 };
  nextDirection = { x: 1, y: 0 };
  score = 0;
  isGameOver = false;
  isPaused = false;
  spawnFood();
  updateScoreUI();
  updateOverlay();
}

function startLoop() {
  if (tickId) clearInterval(tickId);
  tickId = setInterval(gameTick, tickIntervalMs);
  isRunning = true;
}

function stopLoop() {
  if (tickId) clearInterval(tickId);
  tickId = null;
  isRunning = false;
}

function togglePause() {
  if (isGameOver) return;
  isPaused = !isPaused;
  if (isPaused) {
    stopLoop();
  } else {
    startLoop();
  }
  updateOverlay();
  updatePauseButtons();
}

function restartGame() {
  stopLoop();
  initializeGame();
  startLoop();
  updatePauseButtons();
}

function updatePauseButtons() {
  const label = isPaused ? "Resume" : "Pause";
  pauseBtn.textContent = label;
  touchPauseBtn.textContent = label;
}

function updateScoreUI() {
  scoreEl.textContent = String(score);
  highScoreEl.textContent = String(highScore);
}

function updateOverlay() {
  if (isGameOver) {
    overlay.classList.remove("hidden");
    overlayText.textContent = "Game Over";
    return;
  }
  if (isPaused) {
    overlay.classList.remove("hidden");
    overlayText.textContent = "Paused";
  } else {
    overlay.classList.add("hidden");
    overlayText.textContent = "";
  }
}

function spawnFood() {
  while (true) {
    const candidate = {
      x: Math.floor(Math.random() * GRID_COLS),
      y: Math.floor(Math.random() * GRID_ROWS)
    };
    const onSnake = snakeCells.some((c) => c.x === candidate.x && c.y === candidate.y);
    if (!onSnake) {
      foodCell = candidate;
      break;
    }
  }
}

function gameTick() {
  currentDirection = nextDirection;
  const nextHead = {
    x: snakeCells[0].x + currentDirection.x,
    y: snakeCells[0].y + currentDirection.y
  };

  if (isWallCollision(nextHead) || isSelfCollision(nextHead)) {
    handleGameOver();
    return;
  }

  snakeCells.unshift(nextHead);

  if (nextHead.x === foodCell.x && nextHead.y === foodCell.y) {
    score += 1;
    if (score > highScore) {
      highScore = score;
      try { localStorage.setItem("snakeHighScore", String(highScore)); } catch {}
    }
    updateScoreUI();
    maybeIncreaseSpeed();
    spawnFood();
  } else {
    snakeCells.pop();
  }

  draw();
}

function isWallCollision(cell) {
  return cell.x < 0 || cell.y < 0 || cell.x >= GRID_COLS || cell.y >= GRID_ROWS;
}

function isSelfCollision(cell) {
  for (let i = 0; i < snakeCells.length; i++) {
    if (snakeCells[i].x === cell.x && snakeCells[i].y === cell.y) return true;
  }
  return false;
}

function handleGameOver() {
  stopLoop();
  isGameOver = true;
  updateOverlay();
  draw();
}

function maybeIncreaseSpeed() {
  if (score % 5 === 0 && tickIntervalMs > 60) {
    tickIntervalMs -= 6;
    startLoop();
  }
}

function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawFood();
  drawSnake();
}

function drawGrid() {
  context.save();
  context.strokeStyle = "rgba(255,255,255,0.05)";
  context.lineWidth = 1;
  for (let x = 0; x <= GRID_COLS; x++) {
    context.beginPath();
    context.moveTo(x * CELL_SIZE + 0.5, 0);
    context.lineTo(x * CELL_SIZE + 0.5, canvas.height);
    context.stroke();
  }
  for (let y = 0; y <= GRID_ROWS; y++) {
    context.beginPath();
    context.moveTo(0, y * CELL_SIZE + 0.5);
    context.lineTo(canvas.width, y * CELL_SIZE + 0.5);
    context.stroke();
  }
  context.restore();
}

function drawSnake() {
  context.save();
  for (let i = 0; i < snakeCells.length; i++) {
    const cell = snakeCells[i];
    const isHead = i === 0;
    context.fillStyle = isHead ? "#93c5fd" : "#60a5fa";
    context.fillRect(
      cell.x * CELL_SIZE + 1,
      cell.y * CELL_SIZE + 1,
      CELL_SIZE - 2,
      CELL_SIZE - 2
    );
  }
  context.restore();
}

function drawFood() {
  context.save();
  context.fillStyle = "#f59e0b";
  context.beginPath();
  const cx = foodCell.x * CELL_SIZE + CELL_SIZE / 2;
  const cy = foodCell.y * CELL_SIZE + CELL_SIZE / 2;
  const r = CELL_SIZE * 0.35;
  context.arc(cx, cy, r, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function handleDirectionChange(newDirection) {
  const isOpposite = (a, b) => a.x + b.x === 0 && a.y + b.y === 0;
  if (isOpposite(currentDirection, newDirection)) return;
  nextDirection = newDirection;
}

function onKeyDown(event) {
  const code = event.code;
  if (code === "ArrowUp" || code === "KeyW") return handleDirectionChange({ x: 0, y: -1 });
  if (code === "ArrowDown" || code === "KeyS") return handleDirectionChange({ x: 0, y: 1 });
  if (code === "ArrowLeft" || code === "KeyA") return handleDirectionChange({ x: -1, y: 0 });
  if (code === "ArrowRight" || code === "KeyD") return handleDirectionChange({ x: 1, y: 0 });
  if (code === "KeyP" || code === "Space") return togglePause();
  if (code === "KeyR") return restartGame();
}

function setupTouchControls() {
  document.querySelectorAll('.btn[data-dir]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const dir = btn.getAttribute('data-dir');
      if (dir === 'up') handleDirectionChange({ x: 0, y: -1 });
      if (dir === 'down') handleDirectionChange({ x: 0, y: 1 });
      if (dir === 'left') handleDirectionChange({ x: -1, y: 0 });
      if (dir === 'right') handleDirectionChange({ x: 1, y: 0 });
    });
  });

  touchPauseBtn.addEventListener('click', togglePause);
  touchRestartBtn.addEventListener('click', restartGame);
}

pauseBtn.addEventListener('click', togglePause);
restartBtn.addEventListener('click', restartGame);
window.addEventListener('keydown', onKeyDown);

initializeGame();
setupTouchControls();
startLoop();