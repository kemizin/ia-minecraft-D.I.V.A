import time
import random
import numpy as np
from collections import deque
from datetime import datetime

from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import pyautogui
from pynput.keyboard import Key, Controller

# Controle do Minecraft 

teclado = Controller()
teclas_ativas = set()

def segurar_tecla(tecla):
    if tecla not in teclas_ativas:
        teclado.press(tecla)
        teclas_ativas.add(tecla)

def soltar_tecla(tecla):
    if tecla in teclas_ativas:
        teclado.release(tecla)
        teclas_ativas.remove(tecla)

def parar_todas():
    for t in list(teclas_ativas):
        soltar_tecla(t)

def andar_frente(): segurar_tecla('w')
def andar_tras(): segurar_tecla('s')
def andar_esquerda(): segurar_tecla('a')
def andar_direita(): segurar_tecla('d')

def virar_esquerda(): pyautogui.moveRel(-50, 0)
def virar_direita(): pyautogui.moveRel(50, 0)

def pular():
    teclado.press(Key.space)
    time.sleep(0.1)
    teclado.release(Key.space)

def clicar_esquerdo(): pyautogui.click(button='left')
def abrir_inventario():
    teclado.press('e')
    time.sleep(0.1)
    teclado.release('e')

# Captura de Tela

def capturar_tela():
    screenshot = pyautogui.screenshot(region=(0, 40, 800, 600))
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#  Modelo DQN 
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Replay Buffer 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return torch.cat(states), torch.tensor(actions), torch.tensor(rewards), torch.cat(next_states)

    def __len__(self):
        return len(self.buffer)

#  Transformações 
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((84, 84)),
    T.ToTensor()
])

# Parâmetros

acoes = [
    'andar_frente', 'andar_tras', 'andar_esquerda', 'andar_direita',
    'virar_esquerda', 'virar_direita', 'pular', 'clicar_esquerdo', 'abrir_inventario'
]

epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 10000

gamma = 0.99
lr = 1e-4
batch_size = 32
buffer_capacity = 10000
target_update = 1000

#  Funções de Recompensa e Salvamento

def calcula_recompensa(img1, img2):
    diff = np.mean(np.abs(img2.astype(np.float32) - img1.astype(np.float32)))
    return 1.0 if diff > 10 else -0.1

def executa_acao(acao):
    parar_todas()
    if acao == 'andar_frente': andar_frente()
    elif acao == 'andar_tras': andar_tras()
    elif acao == 'andar_esquerda': andar_esquerda()
    elif acao == 'andar_direita': andar_direita()
    elif acao == 'virar_esquerda': virar_esquerda()
    elif acao == 'virar_direita': virar_direita()
    elif acao == 'pular': pular()
    elif acao == 'clicar_esquerdo': clicar_esquerdo()
    elif acao == 'abrir_inventario': abrir_inventario()

def salvar_buffer(buffer, nome="replay_buffer.pt"):
    torch.save(buffer.buffer, nome)

def carregar_buffer(nome="replay_buffer.pt"):
    try:
        dados = torch.load(nome)
        buf = ReplayBuffer(buffer_capacity)
        buf.buffer = dados
        return buf
    except:
        return ReplayBuffer(buffer_capacity)

# função Principal 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    policy_net = DQN(len(acoes)).to(device)
    target_net = DQN(len(acoes)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = carregar_buffer()

    steps_done = 0
    epsilon = epsilon_start

    salvar_intervalo = 300
    ultimo_salvo = time.time()

    estado_atual = capturar_tela()
    estado_atual_proc = transform(estado_atual).unsqueeze(0).to(device)

    try:
        while True:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * max(0, (epsilon_decay - steps_done)) / epsilon_decay
            
            if random.random() < epsilon:
                acao_idx = random.randrange(len(acoes))
            else:
                with torch.no_grad():
                    q_values = policy_net(estado_atual_proc)
                    acao_idx = q_values.argmax().item()

            acao = acoes[acao_idx]
            executa_acao(acao)
            time.sleep(0.2)

            estado_novo = capturar_tela()
            estado_novo_proc = transform(estado_novo).unsqueeze(0).to(device)

            recompensa = calcula_recompensa(estado_atual, estado_novo)
            replay_buffer.push(estado_atual_proc.cpu(), acao_idx, recompensa, estado_novo_proc.cpu())

            estado_atual = estado_novo
            estado_atual_proc = estado_novo_proc
            steps_done += 1

            if len(replay_buffer) >= batch_size:
                estados, acoes_batch, recompensas, estados_next = replay_buffer.sample(batch_size)
                estados, acoes_batch, recompensas, estados_next = estados.to(device), acoes_batch.to(device), recompensas.to(device), estados_next.to(device)

                q_pred = policy_net(estados).gather(1, acoes_batch.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_net(estados_next).max(1)[0]
                q_target = recompensas + gamma * q_next

                loss = F.mse_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if steps_done % 100 == 0:
                    print(f"Passo: {steps_done} | Loss: {loss.item():.4f} | Epsilon: {epsilon:.3f}")

            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if time.time() - ultimo_salvo >= salvar_intervalo:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(policy_net.state_dict(), f"modelo_minecraft_dqn_{timestamp}.pth")
                salvar_buffer(replay_buffer)
                print(f"[SALVO] modelo_minecraft_dqn_{timestamp}.pth")
                ultimo_salvo = time.time()

    except KeyboardInterrupt:
        print("Encerrado pelo usuário.")
    finally:
        parar_todas()

if __name__ == "__main__":
    main()