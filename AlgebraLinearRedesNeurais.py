import numpy as np
import matplotlib.pyplot as plt
import time


class RedeNeuralSimplificada:
    def __init__(self):
        # Inicialização dos pesos e biases da rede neural
        np.random.seed(None)  # Seed aleatória para resultados diferentes a cada execução
        self.W1 = np.random.randn(2, 4) * 0.5  # Pesos da camada de entrada para oculta (2x4)
        self.b1 = np.zeros((1, 4))  # Biases da camada oculta
        self.W2 = np.random.randn(4, 1) * 0.5  # Pesos da camada oculta para saída (4x1)
        self.b2 = np.zeros((1, 1))  # Bias da camada de saída
        self.historico_erro = []  # Armazena o erro a cada época para visualização

    def sigmoid(self, x):
        # Função de ativação sigmoid: transforma qualquer valor entre 0 e 1
        # clip evita overflow (valores muito grandes causam problemas numéricos)
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def derivada_sigmoid(self, x):
        # Derivada da função sigmoid (usada na backpropagation)
        return x * (1 - x)

    def forward(self, X):
        # Propagação para frente: calcula a saída da rede
        self.z1 = np.dot(X, self.W1) + self.b1  # Combinação linear + bias camada oculta
        self.a1 = self.sigmoid(self.z1)  # Aplica função de ativação
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Combinação linear + bias camada saída
        return self.sigmoid(self.z2)  # Saída final da rede

    def backward(self, X, y, output, taxa=0.5):
        # Retropropagação: ajusta os pesos com base no erro
        m = X.shape[0]  # Número de exemplos
        dZ2 = output - y  # Erro na camada de saída
        dW2 = np.dot(self.a1.T, dZ2) / m  # Gradiente dos pesos W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Gradiente dos biases b2
        dZ1 = np.dot(dZ2, self.W2.T) * self.derivada_sigmoid(self.a1)  # Erro na camada oculta
        dW1 = np.dot(X.T, dZ1) / m  # Gradiente dos pesos W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Gradiente dos biases b1

        # Atualiza os pesos e biases usando descida do gradiente
        self.W2 -= taxa * dW2
        self.b2 -= taxa * db2
        self.W1 -= taxa * dW1
        self.b1 -= taxa * db1

        return np.mean(np.abs(dZ2))  # Retorna o erro médio

    def treinar(self, X, y, epochs=5000):
        # Loop principal de treinamento da rede neural
        print("=" * 60)
        print("PROJETO ALGEBRA LINEAR - REDES NEURAIS")
        print("Willian, Pedro e Robson - UFSM")
        print("=" * 60)
        print("TREINAMENTO XOR - Arquitetura: 2-4-1")
        print("-" * 50)

        # Variáveis para parada antecipada
        ultimo_erro = float('inf')
        epochs_sem_melhoria = 0

        for epoch in range(epochs):
            output = self.forward(X)  # Propagação para frente
            erro = self.backward(X, y, output)  # Retropropagação e ajuste de pesos
            self.historico_erro.append(erro)  # Guarda erro para gráfico

            # Verifica se houve melhoria significativa no erro
            if erro < ultimo_erro - 0.0001:
                ultimo_erro = erro
                epochs_sem_melhoria = 0
            else:
                epochs_sem_melhoria += 1

            # Mostra progresso a cada 500 épocas
            if epoch % 500 == 0:
                print(f"Época {epoch:4d}: Erro = {erro:.6f}")

            # Critérios de parada: erro pequeno ou muitas épocas sem melhoria
            if erro < 0.02 or epochs_sem_melhoria > 200:
                if erro < 0.02:
                    print(f"CONVERGENCIA: Época {epoch} (Erro: {erro:.6f})")
                else:
                    print(f"ESTABILIZOU: Época {epoch} (Erro: {erro:.6f})")
                break
        else:
            # Executa se o loop terminar sem break (atingiu máximo de épocas)
            print(f"Limite: {epochs} épocas (Erro: {erro:.6f})")

        self.visualizar_resultado(X, y)  # Mostra resultados visuais

    def visualizar_resultado(self, X, y):
        # Cria gráficos para visualizar o desempenho da rede
        fig = plt.figure(figsize=(15, 5))

        try:
            fig.suptitle(f'RESULTADOS - REDE NEURAL XOR\nConvergiu em {len(self.historico_erro)} épocas',
                         fontsize=14, fontweight='bold')

            # Gráfico 1: Evolução do erro durante o treinamento
            plt.subplot(1, 3, 1)
            if self.historico_erro:
                plt.plot(self.historico_erro, 'r-', linewidth=2)
                plt.title('EVOLUÇÃO DO ERRO')
                plt.xlabel('Época')
                plt.ylabel('Erro')
                plt.grid(True, alpha=0.3)
                if min(self.historico_erro) > 0:
                    plt.yscale('log')  # Escala log para melhor visualização
                if min(self.historico_erro) <= 0.05:
                    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.7)

            # Gráfico 2: Comparação entre valores previstos e esperados
            plt.subplot(1, 3, 2)
            previsoes = self.forward(X)
            posicoes = np.arange(len(X))
            largura = 0.35

            # Barras para valores esperados (azul)
            barras_esperado = plt.bar(posicoes - largura / 2, [v[0] for v in y], largura,
                                      label='Esperado', color='#1f77b4', alpha=0.8,
                                      edgecolor='black')

            # Barras para valores previstos (vermelho)
            barras_previsao = plt.bar(posicoes + largura / 2, [v[0] for v in previsoes], largura,
                                      label='Previsão', color='red', alpha=0.6,
                                      edgecolor='black')

            # Adiciona os valores numéricos acima das barras
            for i, (barra_esp, barra_prev) in enumerate(zip(barras_esperado, barras_previsao)):
                plt.text(barra_esp.get_x() + barra_esp.get_width() / 2,
                         barra_esp.get_height() + 0.03,
                         f'{barra_esp.get_height():.0f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)
                plt.text(barra_prev.get_x() + barra_prev.get_width() / 2,
                         barra_prev.get_height() + 0.03,
                         f'{barra_prev.get_height():.3f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

            plt.title('PREVISÕES vs ESPERADO')
            plt.xlabel('Amostras')
            plt.ylabel('Valor')
            plt.xticks(posicoes, ['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim(0, 1.2)

            # Gráfico 3: Superfície de decisão (como a rede "enxerga" o problema)
            plt.subplot(1, 3, 3)
            x = np.linspace(-0.5, 1.5, 30)
            y_grid = np.linspace(-0.5, 1.5, 30)
            xx, yy = np.meshgrid(x, y_grid)  # Cria malha de pontos
            grid_points = np.c_[xx.ravel(), yy.ravel()]  # Concatena coordenadas
            Z = self.forward(grid_points).reshape(xx.shape)  # Calcula saída para cada ponto

            # Mapa de cores mostrando as regiões de decisão
            contour = plt.contourf(xx, yy, Z, levels=20, alpha=0.7, cmap='RdYlBu_r')
            plt.colorbar(contour, label='Saída')
            plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')

            # Marca os pontos de treinamento no gráfico
            cores = ['darkblue' if val[0] == 0 else 'darkred' for val in y]
            for i, (xi, yi) in enumerate(X):
                plt.scatter(xi, yi, color=cores[i], s=80, edgecolor='black', linewidth=1.5)

            plt.title('SUPERFÍCIE DE DECISÃO')
            plt.xlabel('Entrada x1')
            plt.ylabel('Entrada x2')
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            # Caso ocorra erro na geração dos gráficos
            print(f"Erro na visualização: {e}")
            plt.close(fig)

        self._gerar_relatorio_final(X, y)

    def _gerar_relatorio_final(self, X, y):
        # Gera relatório textual com os resultados finais
        previsoes = self.forward(X)
        print("\n" + "=" * 50)
        print("RELATÓRIO FINAL")
        print("=" * 50)

        acertos = 0
        for i, (entrada, esperado, prev) in enumerate(zip(X, y, previsoes)):
            # Considera correto se previsão e esperado estão do mesmo lado do limiar 0.5
            correto = (prev[0] > 0.5) == (esperado[0] > 0.5)
            acertos += correto
            status = "CORRETO" if correto else "INCORRETO"
            print(f"{entrada} -> {prev[0]:.4f} -> {esperado[0]} -> {status}")

        print("-" * 50)
        print(f"ACURÁCIA: {acertos / len(X) * 100:.1f}%")
        print(f"ERRO FINAL: {self.historico_erro[-1]:.6f}")
        print(f"ÉPOCAS: {len(self.historico_erro)}")


def main():
    # Função principal do programa
    print("REDE NEURAL - PROBLEMA XOR")
    print("Álgebra Linear - UFSM 2025")

    # Dados de treinamento: problema XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    rede = RedeNeuralSimplificada()
    input("\nEnter para iniciar...")

    inicio = time.time()
    rede.treinar(X, y, 5000)  # Treina por até 5000 épocas
    fim = time.time()

    print(f"\nTempo: {fim - inicio:.2f}s")
    print("PROJETO CONCLUÍDO!")


if __name__ == "__main__":
    main()
