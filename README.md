# Calibración de Recompensa Normativa: Modelo Ético Híbrido (CPO-RAG)

Este repositorio contiene el pipeline de datos automatizado y la metodología de evaluación empírica para el documento de investigación: *"Modelo Ético Híbrido vía CMDP con Restricciones Axiomáticas Duras"*. 

El objetivo de este submódulo es proporcionar transparencia total y reproducibilidad algorítmica sobre cómo se calibró la función de recompensa del agente de Aprendizaje por Refuerzo (RL), cuantificando la divergencia entre la inferencia estadística del modelo de lenguaje y las etiquetas de justicia humana.

---

## 1. Fundamento Metodológico

Para evitar la inyección de sesgos arbitrarios en la función de recompensa del CMDP (Proceso de Decisión de Markov Restringido), el sistema requiere una métrica base de equidad. 

Este pipeline extrae un subconjunto *hold-out* del **ETHICS Dataset** (específicamente la partición de prueba del dominio *Justice*, Hendrycks et al., 2020) directamente desde la red de distribución de contenidos (CDN) de Hugging Face. Posteriormente, somete estos dilemas a un proceso de *zero-shot prompting* utilizando `Llama-3-8B-Instruct`.

La divergencia se cuantifica calculando el Error Cuadrático Medio (MSE) entre la etiqueta binaria humana ($Y_i$) y la estimación continua del modelo ($\hat{Y}_i$):

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

Este valor de MSE valida la viabilidad de utilizar LLMs de parámetros reducidos como estimadores dinámicos de utilidad dentro del motor de RL.

---

## 2. Arquitectura del Repositorio

El pipeline está diseñado bajo principios de mínima fricción y dependencias aisladas. No requiere descarga manual de datasets; los datos fluyen dinámicamente a la memoria durante la ejecución.

* `calibrate_reward_mse.py`: Pipeline principal de orquestación, extracción y evaluación.
* `requirements.txt`: Declaración de dependencias (flexibilizadas para máxima compatibilidad multiplataforma).
* `.env.example`: Plantilla para la configuración de credenciales de la API de inferencia.
* `data/`: Directorio autogenerado donde se almacenan los artefactos de salida (CSV) tras cada ejecución para su auditoría.

---

## 3. Guía de Reproducibilidad (Entorno y Ejecución)

Para garantizar la integridad del entorno de ejecución e impedir conflictos de dependencias a nivel de sistema operativo, se requiere el uso de entornos virtuales. 

### 3.1. Requisitos Previos del Sistema
* **Python 3.9+** * Una clave de API gratuita de [Groq Cloud](https://console.groq.com/) (utilizada para inferencia de ultra-baja latencia con Llama-3).

### 3.2. Configuración del Entorno Aislado

Clona el repositorio y navega al directorio raíz:
```bash
git clone [https://github.com/TU-USUARIO/ethics-cpo-calibration.git](https://github.com/TU-USUARIO/ethics-cpo-calibration.git)
cd ethics-cpo-calibration