"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScanLine, UserCheck, UserPlus, Shield, Camera } from "lucide-react";
import { PhotoUpload } from "@/app/match/components/image_uploader";
import { setToken, setUser } from "@/app/auth";
import { useRouter } from "next/navigation";

interface User {
  username: string;
}

type AccessState =
  | "scanning"
  | "analyzing"
  | "matched"
  | "not-found"
  | "registering";

const Access = () => {
  const [selectedPhoto, setSelectedPhoto] = useState<File | null>(null);
  const [accessState, setAccessState] = useState<AccessState>("scanning");
  const [matchedUser, setMatchedUser] = useState<User | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [registrationData, setRegistrationData] = useState({
    username: "",
    name: "",
    password: "",
    email: "",
  });

  const router = useRouter();

  const handlePhotoSelect = (file: File) => {
    setSelectedPhoto(file);
  };

  const handleRemovePhoto = () => {
    setSelectedPhoto(null);
    setAccessState("scanning");
    setMatchedUser(null);
  };

  // Login con imagen
  const handleScanFace = async () => {
    if (!selectedPhoto) return;

    setIsAnalyzing(true);
    setAccessState("analyzing");

    await new Promise((resolve) => setTimeout(resolve, 1500));

    const formData = new FormData();
    formData.append("image", selectedPhoto);

    try {
      const response = await fetch("http://localhost:5000/api/auth/login", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      const success = data.token;

      if (success) {
        setToken(data.token);
        setUser(data.user);
        setMatchedUser(data.user);
        setAccessState("matched");

        setTimeout(() => {
          router.push("/dashboard");
        }, 5000);
      } else {
        setAccessState("not-found");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Registro de nuevo usuario
  const handleSubmitRegistration = async () => {
    try {
      if (!selectedPhoto) return;

      setIsAnalyzing(true);

      await new Promise((resolve) => setTimeout(resolve, 1500));

      const formData = new FormData();
      formData.append("image", selectedPhoto);
      formData.append("username", registrationData.username);
      formData.append("name", registrationData.name);
      formData.append("email", registrationData.email);
      formData.append("password", registrationData.password);

      const response = await fetch(
        "http://localhost:5000/api/auth/register",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();
      const success = data.token;

      if (success) {
        setToken(data.token);
        setUser(data.user);
        setMatchedUser(data.user);
        setAccessState("matched");

        setTimeout(() => router.push("/dashboard"), 5000);
      } else {
        setAccessState("not-found");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Reinicia los estados
  const reset = () => {
    setSelectedPhoto(null);
    setAccessState("scanning");
    setMatchedUser(null);
    setRegistrationData({
      name: "",
      email: "",
      username: "",
      password: "",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-6 mb-12"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center justify-center w-24 h-24 bg-primary/10 rounded-full border border-primary/20"
          >
            <Shield className="w-12 h-12 text-primary" />
          </motion.div>

          <div className="space-y-4">
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Control de Acceso
            </h1>

            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Acceso seguro con reconocimiento facial para usuarios
              autorizados
            </p>
          </div>
        </motion.div>

        <div className="max-w-2xl mx-auto">
          <AnimatePresence mode="wait">
            {accessState === "scanning" && (
              <motion.div
                key="scanning"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle className="text-center flex items-center justify-center gap-2">
                      <Camera className="h-6 w-6" />
                      Escaner de reconocimiento facial
                    </CardTitle>
                  </CardHeader>

                  <CardContent className="space-y-6">
                    <PhotoUpload
                      onPhotoSelect={handlePhotoSelect}
                      selectedPhoto={selectedPhoto}
                      onRemovePhoto={handleRemovePhoto}
                    />

                    {selectedPhoto && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-center"
                      >
                        <Button
                          size="lg"
                          onClick={handleScanFace}
                          className="min-w-[200px]"
                        >
                          <ScanLine className="h-5 w-5" />
                          Escanear
                        </Button>
                      </motion.div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {accessState === "analyzing" && (
              <motion.div
                key="analyzing"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <Card>
                  <CardContent className="p-12 text-center space-y-6">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "linear",
                      }}
                      className="w-16 h-16 mx-auto border-4 border-primary border-t-transparent rounded-full"
                    />
                    <h3 className="text-xl font-semibold">
                      Analizando características faciales...
                    </h3>
                    <p className="text-muted-foreground">
                      Porfavor aguarde.
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {accessState === "matched" && matchedUser && (
              <motion.div
                key="matched"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="border-green-200 bg-green-50/50 dark:border-green-800 dark:bg-green-950/50">
                  <CardContent className="p-12 text-center space-y-6">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{
                        delay: 0.2,
                        type: "spring",
                        stiffness: 200,
                      }}
                      className="w-20 h-20 mx-auto bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center"
                    >
                      <UserCheck className="w-10 h-10 text-green-600 dark:text-green-400" />
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="space-y-2"
                    >
                      <h2 className="text-3xl font-bold text-green-700 dark:text-green-300">
                        Bienvenido, {matchedUser.username}
                      </h2>
                      <p className="text-green-600 dark:text-green-400">
                        Serás redirigido automáticamente, aguarda.
                      </p>
                    </motion.div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {accessState === "not-found" && (
              <motion.div
                key="not-found"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="border-orange-200 bg-orange-50/50 dark:border-orange-800 dark:bg-orange-950/50">
                  <CardContent className="p-12 text-center space-y-6">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{
                        delay: 0.2,
                        type: "spring",
                        stiffness: 200,
                      }}
                      className="w-20 h-20 mx-auto bg-orange-100 dark:bg-orange-900 rounded-full flex items-center justify-center"
                    >
                      <UserPlus className="w-10 h-10 text-orange-600 dark:text-orange-400" />
                    </motion.div>

                    <div className="space-y-2">
                      <h2 className="text-2xl font-bold text-orange-700 dark:text-orange-300">
                        No se encontró ninguna coincidencia
                      </h2>
                      <p className="text-orange-600 dark:text-orange-400">
                        No estás registrado. ¿Quieres crear una cuenta?
                      </p>
                    </div>

                    <div className="flex gap-4 justify-center">
                      <Button onClick={() => setAccessState("registering")}>
                        Registrarme
                      </Button>
                      <Button variant="outline" onClick={reset}>
                        Volver a intentarlo
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {accessState === "registering" && (
              <motion.div
                key="registering"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle className="text-center flex items-center justify-center gap-2">
                      <UserPlus className="h-6 w-6" />
                      Crear cuenta
                    </CardTitle>
                  </CardHeader>

                  <CardContent className="space-y-6">
                    <PhotoUpload
                      onPhotoSelect={handlePhotoSelect}
                      selectedPhoto={selectedPhoto}
                      onRemovePhoto={handleRemovePhoto}
                    />

                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="name">Nombre completo</Label>
                        <Input
                          id="name"
                          value={registrationData.name}
                          onChange={(e) =>
                            setRegistrationData((prev) => ({
                              ...prev,
                              name: e.target.value,
                            }))
                          }
                          placeholder="John Doe"
                        />
                      </div>

                      <div>
                        <Label htmlFor="username">Nombre de usuario</Label>
                        <Input
                          id="username"
                          value={registrationData.username}
                          onChange={(e) =>
                            setRegistrationData((prev) => ({
                              ...prev,
                              username: e.target.value,
                            }))
                          }
                          placeholder="john123"
                        />
                      </div>

                      <div>
                        <Label htmlFor="email">Correo electrónico</Label>
                        <Input
                          id="email"
                          type="email"
                          value={registrationData.email}
                          onChange={(e) =>
                            setRegistrationData((prev) => ({
                              ...prev,
                              email: e.target.value,
                            }))
                          }
                          placeholder="email@ejemplo.com"
                        />
                      </div>

                      <div>
                        <Label htmlFor="password">Contraseña</Label>
                        <Input
                          id="password"
                          type="password"
                          value={registrationData.password}
                          onChange={(e) =>
                            setRegistrationData((prev) => ({
                              ...prev,
                              password: e.target.value,
                            }))
                          }
                          placeholder="********"
                        />
                      </div>
                    </div>

                    <div className="flex gap-4 justify-center pt-4">
                      <Button
                        onClick={handleSubmitRegistration}
                        disabled={
                          !selectedPhoto ||
                          !registrationData.name ||
                          !registrationData.username ||
                          !registrationData.email ||
                          !registrationData.password
                        }
                      >
                        Crear cuenta
                      </Button>

                      <Button
                        variant="outline"
                        onClick={() => setAccessState("not-found")}
                      >
                        Volver
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default Access;
