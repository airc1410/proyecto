import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def extraer_fecha_del_nombre_archivo(nombre_archivo):
    coincidencia = re.search(r'\d{4}\.\d{2}\.\d{2}', nombre_archivo)
    if coincidencia:
        año, mes, día = coincidencia.group(0).split('.')
        return int(año), int(mes), int(día)
    return None, None, None

def obtener_rango_columnas(rango):
    columnas = []
    partes = rango.split(':')
    if len(partes) == 2:
        start, end = partes
        for col in range(ord(start.upper()), ord(end.upper()) + 1):
            columnas.append(chr(col))
    return columnas

def convertir_columnas_a_indices(df, columnas):
    max_columna = len(df.columns)
    indices_columnas = []
    for columna in columnas:
        indice = ord(columna) - ord('A')
        if indice < max_columna:
            indices_columnas.append(indice)
    return indices_columnas

def procesar_archivos(archivos, rango_columnas, fila_inicio):
    todos_datos = []
    columnas = obtener_rango_columnas(rango_columnas)

    for archivo in archivos:
        try:
            st.write(f"Procesando archivo: {archivo.name}")
            
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo, skiprows=fila_inicio-1)
            else:
                df_temporal = pd.read_excel(archivo, nrows=1)
                st.write(f"Columnas disponibles en {archivo.name}: {list(df_temporal.columns)}")
                indices_columnas = convertir_columnas_a_indices(df_temporal, columnas)
                st.write(f"Índices de columnas seleccionadas para {archivo.name}: {indices_columnas}")

                if not indices_columnas:
                    st.error(f"Error al procesar el archivo {archivo.name}: Las columnas especificadas están fuera del rango permitido.")
                    continue

                df = pd.read_excel(archivo, skiprows=fila_inicio-1, usecols=indices_columnas)
            
            st.write(f"Shape del DataFrame: {df.shape}")
            st.write(f"Primeras filas del DataFrame:")
            st.write(df.head())

            año, mes, día = extraer_fecha_del_nombre_archivo(archivo.name)
            if año is not None:
                df['AÑO'] = año
                df['MES'] = mes
                df['DÍA'] = día
            else:
                st.warning(f"No se pudo extraer la fecha del nombre del archivo: {archivo.name}")
            
            if not df.empty:
                todos_datos.append(df)
            else:
                st.warning(f"El DataFrame para {archivo.name} está vacío.")

        except ValueError as ve:
            st.error(f"Error al procesar el archivo {archivo.name}: {ve}")
        except Exception as e:
            st.error(f"Error inesperado al procesar el archivo {archivo.name}: {e}")

    if todos_datos:
        df_combinado = pd.concat(todos_datos, ignore_index=True)
        return df_combinado
    else:
        st.error("No se encontraron datos válidos para procesar.")
        return pd.DataFrame()

def guardar_en_excel(df, archivo_salida):
    try:
        df.to_excel(archivo_salida, index=False)
    except Exception as e:
        st.error(f"Error al guardar el archivo {archivo_salida}: {e}")

def generar_grafico_regresion(df):
    try:
        if df.empty:
            st.warning("No hay datos para graficar.")
            return None

        columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
        if len(columnas_numericas) >= 2:
            x_col = columnas_numericas[0]
            y_col = columnas_numericas[1]

            X = df[[x_col]].values
            y = df[y_col].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            fig, ax = plt.subplots()
            sns.regplot(x=df[x_col], y=df[y_col], ax=ax, line_kws={"color": "red"})
            ax.set_title(f'Regresión Lineal\n$R^2$ = {r2:.2f}')
            st.pyplot(fig)
            return fig
        else:
            st.warning("El DataFrame debe tener al menos dos columnas numéricas para generar un gráfico de regresión.")
            return None
    except Exception as e:
        st.error(f"Error al generar el gráfico de regresión: {e}")
        return None

def main():
    st.title("Proceso ETL y Análisis de Datos")

    if 'df_final' not in st.session_state:
        st.session_state.df_final = pd.DataFrame()

    if 'figuras' not in st.session_state:
        st.session_state.figuras = []

    st.header("Carga y Procesamiento de Archivos")
    archivos_subidos = st.file_uploader("Seleccionar Archivos Excel o CSV", accept_multiple_files=True, type=["xls", "xlsx", "csv"])
    columnas_seleccionadas = st.text_input("Rango de Columnas (ej. A:Z)")
    fila_inicio = st.number_input("Fila Inicial", min_value=1, step=1, value=1)

    archivo_salida = st.text_input("Nombre del archivo de salida (sin extensión)", value="Out")

    if st.button("Ejecutar ETL"):
        if archivos_subidos and columnas_seleccionadas:
            df_final = procesar_archivos(archivos_subidos, columnas_seleccionadas, fila_inicio)
            if not df_final.empty:
                st.session_state.df_final = df_final
                archivo_salida_completo = f"{archivo_salida}.xlsx"
                guardar_en_excel(df_final, archivo_salida_completo)
                st.success(f"Datos procesados y guardados exitosamente como {archivo_salida_completo}")
                st.dataframe(df_final)
                st.write(f"Shape del DataFrame final: {df_final.shape}")
                st.session_state.figuras = []  # Limpiar gráficos antiguos

            else:
                st.warning("No se pudieron procesar los datos. Verifique los archivos de entrada y los parámetros.")
        else:
            st.warning("Por favor, suba archivos y especifique el rango de columnas.")

    st.header("Análisis de Datos")
    if not st.session_state.df_final.empty:
        st.write("Generando gráfico de regresión lineal...")
        fig_regresion = generar_grafico_regresion(st.session_state.df_final)
        if fig_regresion:
            st.session_state.figuras.append(('regresion', fig_regresion))

        if st.button("Guardar Gráfico de Regresión Lineal"):
            ruta_guardar = os.path.expanduser("~/Documents/python/graficos")
            if not os.path.exists(ruta_guardar):
                os.makedirs(ruta_guardar)

            for nombre, figura in st.session_state.figuras:
                if nombre == 'regresion':
                    guardar_grafico(figura, os.path.join(ruta_guardar, f"{nombre}.png"))

    else:
        st.warning("No hay datos para analizar. Por favor, ejecute el proceso ETL primero.")
        st.write("Estado actual del DataFrame:")
        st.write(st.session_state.df_final)

if __name__ == "__main__":
    main()
