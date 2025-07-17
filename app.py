import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Análisis Multivariado - Concreto", layout="wide")

def estandarizar_datos(X):
    """
    Estandariza los datos (z-score)
    """
    media = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - media) / std
    return X_std, media, std

def regresion_lineal_multiple(X, y):
    """
    Implementación de regresión lineal múltiple usando numpy
    """
    # Añadir columna de unos para el intercepto
    X_con_intercepto = np.column_stack([np.ones(X.shape[0]), X])
    
    # Calcular coeficientes usando la ecuación normal: β = (X^T X)^(-1) X^T y
    try:
        coeficientes = np.linalg.solve(X_con_intercepto.T @ X_con_intercepto, X_con_intercepto.T @ y)
        intercepto = coeficientes[0]
        coefs = coeficientes[1:]
    except np.linalg.LinAlgError:
        # Si la matriz es singular, usar pseudoinversa
        coeficientes = np.linalg.pinv(X_con_intercepto.T @ X_con_intercepto) @ X_con_intercepto.T @ y
        intercepto = coeficientes[0]
        coefs = coeficientes[1:]
    
    return coefs, intercepto

def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R²
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def calcular_rmse(y_real, y_pred):
    """
    Calcula el error cuadrático medio (RMSE)
    """
    mse = np.mean((y_real - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def predecir(X, coefs, intercepto):
    """
    Realiza predicciones usando el modelo de regresión
    """
    return X @ coefs + intercepto

def calcular_resistencia_estimada(cemento, agua, agregado_grueso, agregado_fino, aditivo, edad):
    """
    Modelo simplificado para estimar resistencia del concreto
    Basado en principios de ingeniería civil
    """
    # Relación agua/cemento
    relacion_ac = agua / cemento if cemento > 0 else 0
    
    # Fórmula empírica simplificada
    resistencia = (
        28 * (1 - 0.5 * relacion_ac) * 
        (1 + 0.1 * aditivo/100) * 
        (1 + 0.05 * agregado_grueso/1000) * 
        (1 + 0.03 * agregado_fino/1000) * 
        (edad / 28) ** 0.5
    )
    
    return max(resistencia, 0)

def generar_datos_muestra():
    """
    Genera datos de muestra para demostración
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'cemento_kg_m3': np.random.normal(350, 50, n_samples),
        'agua_kg_m3': np.random.normal(180, 30, n_samples),
        'agregado_grueso_kg_m3': np.random.normal(1200, 100, n_samples),
        'agregado_fino_kg_m3': np.random.normal(800, 80, n_samples),
        'aditivo_porcentaje': np.random.normal(2, 0.5, n_samples),
        'edad_dias': np.random.choice([7, 14, 21, 28], n_samples)
    }
    
    df = pd.DataFrame(data)
    df['resistencia_mpa'] = df.apply(
        lambda row: calcular_resistencia_estimada(
            row['cemento_kg_m3'], row['agua_kg_m3'], 
            row['agregado_grueso_kg_m3'], row['agregado_fino_kg_m3'],
            row['aditivo_porcentaje'], row['edad_dias']
        ) + np.random.normal(0, 2), axis=1
    )
    
    return df

def calcular_estadisticas_descriptivas(df):
    """
    Calcula estadísticas descriptivas personalizadas
    """
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'Media': df[col].mean(),
            'Mediana': df[col].median(),
            'Desv. Std': df[col].std(),
            'Mínimo': df[col].min(),
            'Máximo': df[col].max(),
            'Q1': df[col].quantile(0.25),
            'Q3': df[col].quantile(0.75)
        }
    return pd.DataFrame(stats).T

def main():
    st.title("🏗️ Análisis Multivariado de Resistencia del Concreto")
    st.markdown("Análisis de factores que influyen en la resistencia a compresión del concreto")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración del Análisis")
    
    # Opciones de datos
    opcion_datos = st.sidebar.selectbox(
        "Fuente de datos:",
        ["Datos de muestra", "Ingresar datos manualmente", "Cargar archivo CSV"]
    )
    
    df = None
    
    if opcion_datos == "Datos de muestra":
        df = generar_datos_muestra()
        st.sidebar.success("✅ Datos de muestra cargados")
        
    elif opcion_datos == "Ingresar datos manualmente":
        st.sidebar.subheader("📝 Ingreso Manual de Datos")
        
        # Crear formulario para ingreso manual
        with st.sidebar.form("datos_manuales"):
            st.write("**Dosificación por m³ de concreto:**")
            cemento = st.number_input("Cemento (kg/m³)", min_value=200.0, max_value=500.0, value=350.0)
            agua = st.number_input("Agua (kg/m³)", min_value=120.0, max_value=250.0, value=180.0)
            agregado_grueso = st.number_input("Agregado grueso (kg/m³)", min_value=800.0, max_value=1500.0, value=1200.0)
            agregado_fino = st.number_input("Agregado fino (kg/m³)", min_value=600.0, max_value=1000.0, value=800.0)
            aditivo = st.number_input("Aditivo (%)", min_value=0.0, max_value=5.0, value=2.0)
            edad = st.selectbox("Edad (días)", [7, 14, 21, 28], index=3)
            
            submitted = st.form_submit_button("Calcular Resistencia")
            
            if submitted:
                resistencia = calcular_resistencia_estimada(cemento, agua, agregado_grueso, agregado_fino, aditivo, edad)
                st.sidebar.success(f"Resistencia estimada: {resistencia:.2f} MPa")
                
                # Crear DataFrame con un solo dato
                df = pd.DataFrame({
                    'cemento_kg_m3': [cemento],
                    'agua_kg_m3': [agua],
                    'agregado_grueso_kg_m3': [agregado_grueso],
                    'agregado_fino_kg_m3': [agregado_fino],
                    'aditivo_porcentaje': [aditivo],
                    'edad_dias': [edad],
                    'resistencia_mpa': [resistencia]
                })
    
    elif opcion_datos == "Cargar archivo CSV":
        uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Archivo CSV cargado exitosamente")
            except Exception as e:
                st.sidebar.error(f"Error al cargar el archivo: {e}")
    
    # Análisis principal
    if df is not None and not df.empty:
        
        # Estadísticas descriptivas
        st.header("📊 Estadísticas Descriptivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resumen Estadístico")
            stats_df = calcular_estadisticas_descriptivas(df)
            st.dataframe(stats_df)
        
        with col2:
            st.subheader("Información del Dataset")
            st.write(f"**Número de muestras:** {len(df)}")
            st.write(f"**Variables:** {df.columns.tolist()}")
            
            # Relación agua/cemento
            if 'agua_kg_m3' in df.columns and 'cemento_kg_m3' in df.columns:
                df['relacion_ac'] = df['agua_kg_m3'] / df['cemento_kg_m3']
                relacion_ac_promedio = df['relacion_ac'].mean()
                st.write(f"**Relación A/C promedio:** {relacion_ac_promedio:.3f}")
        
        # Matriz de correlación
        st.header("🔗 Matriz de Correlación")
        
        # Seleccionar variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Matriz de Correlación - Variables del Concreto')
            st.pyplot(fig_corr)
        
        # Análisis de regresión múltiple
        st.header("📈 Análisis de Regresión Múltiple")
        
        if 'resistencia_mpa' in df.columns and len(df) > 1:
            # Preparar datos
            X_cols = [col for col in numeric_cols if col != 'resistencia_mpa']
            if X_cols and len(X_cols) > 0:
                X = df[X_cols].values
                y = df['resistencia_mpa'].values
                
                # Estandarizar datos
                X_std, media_X, std_X = estandarizar_datos(X)
                
                # Entrenar modelo
                try:
                    coefs, intercepto = regresion_lineal_multiple(X_std, y)
                    
                    # Predicciones
                    y_pred = predecir(X_std, coefs, intercepto)
                    
                    # Métricas
                    r2 = calcular_r2(y, y_pred)
                    rmse = calcular_rmse(y, y_pred)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Métricas del Modelo")
                        st.write(f"**R² Score:** {r2:.4f}")
                        st.write(f"**RMSE:** {rmse:.4f} MPa")
                        
                        # Coeficientes
                        st.subheader("Coeficientes del Modelo")
                        coef_df = pd.DataFrame({
                            'Variable': X_cols,
                            'Coeficiente': coefs,
                            'Coef_Abs': np.abs(coefs)
                        }).sort_values('Coef_Abs', ascending=False)
                        st.dataframe(coef_df[['Variable', 'Coeficiente']])
                    
                    with col2:
                        # Gráfico de predicción vs real
                        fig_pred = plt.figure(figsize=(8, 6))
                        plt.scatter(y, y_pred, alpha=0.7)
                        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                        plt.xlabel('Resistencia Real (MPa)')
                        plt.ylabel('Resistencia Predicha (MPa)')
                        plt.title('Predicción vs Valores Reales')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig_pred)
                        
                        # Residuos
                        residuos = y - y_pred
                        st.subheader("Análisis de Residuos")
                        st.write(f"**Media de residuos:** {np.mean(residuos):.4f}")
                        st.write(f"**Desv. std residuos:** {np.std(residuos):.4f}")
                        
                except Exception as e:
                    st.error(f"Error en el análisis de regresión: {e}")
        
        # Visualizaciones interactivas
        st.header("📊 Visualizaciones Interactivas")
        
        # Opciones de gráficos
        tipo_grafico = st.selectbox(
            "Selecciona el tipo de gráfico:",
            ["Scatter Plot", "Box Plot", "Histograma", "Distribución por Edad", "Análisis 3D"]
        )
        
        if tipo_grafico == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Variable X:", numeric_cols)
            with col2:
                y_var = st.selectbox("Variable Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            if x_var and y_var:
                fig = px.scatter(df, x=x_var, y=y_var, 
                               title=f'{y_var} vs {x_var}',
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
        
        elif tipo_grafico == "Box Plot":
            if 'edad_dias' in df.columns and 'resistencia_mpa' in df.columns:
                fig = px.box(df, x='edad_dias', y='resistencia_mpa',
                            title='Distribución de Resistencia por Edad')
                st.plotly_chart(fig, use_container_width=True)
        
        elif tipo_grafico == "Histograma":
            var_hist = st.selectbox("Variable para histograma:", numeric_cols)
            if var_hist:
                fig = px.histogram(df, x=var_hist, nbins=20,
                                 title=f'Distribución de {var_hist}')
                st.plotly_chart(fig, use_container_width=True)
        
        elif tipo_grafico == "Distribución por Edad":
            if 'edad_dias' in df.columns and 'resistencia_mpa' in df.columns:
                fig = px.violin(df, x='edad_dias', y='resistencia_mpa',
                               title='Distribución de Resistencia por Edad del Concreto')
                st.plotly_chart(fig, use_container_width=True)
        
        elif tipo_grafico == "Análisis 3D":
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_var = st.selectbox("Variable X:", numeric_cols, key="3d_x")
                with col2:
                    y_var = st.selectbox("Variable Y:", numeric_cols, key="3d_y")
                with col3:
                    z_var = st.selectbox("Variable Z:", numeric_cols, key="3d_z")
                
                if x_var and y_var and z_var:
                    fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var,
                                       title=f'Análisis 3D: {x_var} vs {y_var} vs {z_var}')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de importancia de variables
        st.header("📊 Análisis de Importancia de Variables")
        
        if 'resistencia_mpa' in df.columns and len(numeric_cols) > 1:
            correlaciones = df[numeric_cols].corr()['resistencia_mpa'].drop('resistencia_mpa')
            correlaciones_abs = correlaciones.abs().sort_values(ascending=False)
            
            fig_imp = plt.figure(figsize=(10, 6))
            correlaciones_abs.plot(kind='bar')
            plt.title('Importancia de Variables (Correlación Absoluta con Resistencia)')
            plt.ylabel('Correlación Absoluta')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_imp)
        
        # Predictor interactivo
        st.header("🎯 Predictor de Resistencia")
        
        if 'resistencia_mpa' in df.columns and len(df) > 1:
            st.subheader("Ingresa los parámetros para predecir la resistencia:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_cemento = st.slider("Cemento (kg/m³)", 200.0, 500.0, 350.0)
                pred_agua = st.slider("Agua (kg/m³)", 120.0, 250.0, 180.0)
            
            with col2:
                pred_ag_grueso = st.slider("Agregado grueso (kg/m³)", 800.0, 1500.0, 1200.0)
                pred_ag_fino = st.slider("Agregado fino (kg/m³)", 600.0, 1000.0, 800.0)
            
            with col3:
                pred_aditivo = st.slider("Aditivo (%)", 0.0, 5.0, 2.0)
                pred_edad = st.select_slider("Edad (días)", options=[7, 14, 21, 28], value=28)
            
            # Calcular predicción
            resistencia_pred = calcular_resistencia_estimada(
                pred_cemento, pred_agua, pred_ag_grueso, 
                pred_ag_fino, pred_aditivo, pred_edad
            )
            
            # Mostrar resultado
            relacion_ac = pred_agua / pred_cemento
            
            st.subheader("Resultados de la Predicción:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Resistencia Estimada", f"{resistencia_pred:.2f} MPa")
            with col2:
                st.metric("Relación A/C", f"{relacion_ac:.3f}")
            with col3:
                calidad = "Excelente" if resistencia_pred > 35 else "Buena" if resistencia_pred > 25 else "Regular"
                st.metric("Calidad", calidad)
            
            # Análisis de sensibilidad
            st.subheader("Análisis de Sensibilidad")
            
            # Variación del cemento
            cemento_range = np.linspace(250, 450, 20)
            resistencia_cemento = [calcular_resistencia_estimada(c, pred_agua, pred_ag_grueso, 
                                                               pred_ag_fino, pred_aditivo, pred_edad) 
                                 for c in cemento_range]
            
            fig_sens = plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(cemento_range, resistencia_cemento, 'b-', linewidth=2)
            plt.xlabel('Cemento (kg/m³)')
            plt.ylabel('Resistencia (MPa)')
            plt.title('Sensibilidad al Cemento')
            plt.grid(True, alpha=0.3)
            
            # Variación del agua
            agua_range = np.linspace(140, 220, 20)
            resistencia_agua = [calcular_resistencia_estimada(pred_cemento, a, pred_ag_grueso, 
                                                            pred_ag_fino, pred_aditivo, pred_edad) 
                              for a in agua_range]
            
            plt.subplot(1, 3, 2)
            plt.plot(agua_range, resistencia_agua, 'r-', linewidth=2)
            plt.xlabel('Agua (kg/m³)')
            plt.ylabel('Resistencia (MPa)')
            plt.title('Sensibilidad al Agua')
            plt.grid(True, alpha=0.3)
            
            # Variación de la edad
            edad_range = np.array([7, 14, 21, 28])
            resistencia_edad = [calcular_resistencia_estimada(pred_cemento, pred_agua, pred_ag_grueso, 
                                                            pred_ag_fino, pred_aditivo, e) 
                              for e in edad_range]
            
            plt.subplot(1, 3, 3)
            plt.plot(edad_range, resistencia_edad, 'g-o', linewidth=2)
            plt.xlabel('Edad (días)')
            plt.ylabel('Resistencia (MPa)')
            plt.title('Sensibilidad a la Edad')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_sens)
        
        # Datos tabulares
        st.header("📋 Datos Completos")
        st.dataframe(df, use_container_width=True)
        
        # Descarga de datos
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar datos como CSV",
            data=csv,
            file_name='analisis_concreto.csv',
            mime='text/csv'
        )
    
    else:
        st.warning("⚠️ Por favor, carga o ingresa datos para comenzar el análisis.")
    
    # Información técnica
    st.header("ℹ️ Información Técnica")
    
    with st.expander("📚 Fundamentos del Análisis"):
        st.markdown("""
        ### Variables Analizadas:
        - **Cemento (kg/m³)**: Principal componente aglutinante
        - **Agua (kg/m³)**: Necesaria para la hidratación del cemento
        - **Agregado grueso (kg/m³)**: Proporciona resistencia estructural
        - **Agregado fino (kg/m³)**: Mejora la trabajabilidad
        - **Aditivo (%)**: Mejora propiedades específicas
        - **Edad (días)**: Tiempo de curado
        
        ### Factores Clave:
        - **Relación A/C**: Factor más influyente en la resistencia
        - **Tiempo de curado**: La resistencia aumenta con el tiempo
        - **Calidad de materiales**: Influye significativamente
        
        ### Metodología:
        - **Regresión lineal múltiple**: Implementada con numpy
        - **Análisis de correlación**: Matriz de Pearson
        - **Estandarización**: Z-score normalización
        - **Análisis de sensibilidad**: Variación paramétrica
        """)

if __name__ == "__main__":
    main()